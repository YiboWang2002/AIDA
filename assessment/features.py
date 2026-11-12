"""Temporal pooling of COVAREP, formants and CLNF descriptors."""

import csv
import math

import numpy as np


def _number(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return float('nan')


_FEATURE_CHUNK_ROWS = 8192


def _feature_numeric_chunk(lines, delimiter, expected_columns, indices, headerless_width, row_start):
    """Fast bounded numeric parsing, with the original CSV/cell rules as fallback."""
    import warnings
    # Counting fields prevents a short row and a long row cancelling in reshape.
    simple = all('"' not in line and line.count(delimiter) + 1 == expected_columns for line in lines)
    if simple and max(indices) < expected_columns and (headerless_width is None or expected_columns == headerless_width):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('error')
                values = np.fromstring(delimiter.join(line.strip() for line in lines), sep=delimiter, dtype=np.float64)
            if values.size == len(lines) * expected_columns:
                return values.reshape(len(lines), expected_columns)[:, indices]
        except (ValueError, Warning):
            pass
    values = np.empty((len(lines), len(indices)), dtype=np.float64)
    for offset, line in enumerate(lines):
        row = next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))
        row_index = row_start + offset
        if headerless_width is not None and len(row) != headerless_width:
            raise ValueError('exact_headerless_official_feature_width_required:' + str(row_index))
        if max(indices) >= len(row):
            raise ValueError('ragged_feature_row:' + str(row_index))
        values[offset] = [_number(row[i]) for i in indices]
    return values


def stream_aggregate(binary, spec, turns, width):
    """Pool finite measurements over the speech intervals of each turn."""
    iterator = (line.decode('utf-8-sig') for line in binary)
    first = next(iterator, None)
    if first is None:
        raise ValueError('empty_feature_member')
    delimiter = spec.get('delimiter') or ('\t' if '\t' in first else ',')
    parse = lambda line: next(csv.reader([line], delimiter=delimiter, skipinitialspace=True))
    first_row = [v.strip() for v in parse(first)]
    header = spec.get('header', 'auto')
    if header == 'auto':
        header = any(not math.isfinite(_number(v)) and v.lower() not in ('nan', 'inf', '-inf') for v in first_row)
    names = [v.strip().lower() for v in first_row] if header else None
    def index(key):
        if isinstance(key, int):
            return key
        if names is None or key.lower() not in names:
            raise ValueError('feature_column_missing:' + str(key))
        return names.index(key.lower())
    time_index = index(spec['timestamp']) if spec.get('timestamp') is not None else None
    quality_indices = {k: index(spec[k]) for k in ('confidence', 'success') if spec.get(k) is not None}
    if 'columns' in spec:
        columns = [index(k) for k in spec['columns']]
    elif names is not None:
        excluded = {time_index, *quality_indices.values()}
        excluded.update(i for i, name in enumerate(names) if name in ('frame', 'face_id'))
        columns = [i for i in range(len(names)) if i not in excluded]
    else:
        columns = list(range(width))
    if len(columns) != width or len(set(columns)) != width:
        raise ValueError('exact_official_feature_width_required:' + str((width, columns)))
    count = np.zeros((len(turns), width), dtype=np.int64)
    sums = np.zeros_like(count, dtype=np.float64)
    squares = np.zeros_like(sums)
    rows, last_time = 0, None
    # Merge per-owner intervals so even overlapping source intervals count once.
    intervals = []
    for turn in turns:
        union = []
        for start, stop in sorted(turn['intervals']):
            if union and start <= union[-1][1]:
                union[-1][1] = max(union[-1][1], stop)
            else:
                union.append([start, stop])
        intervals.append(np.asarray(union, dtype=np.float64).reshape(-1, 2))
    selected = columns + list(quality_indices.values()) + ([time_index] if time_index is not None else [])
    quality_positions = {key: width + i for i, key in enumerate(quality_indices)}
    minimum_confidence = float(spec.get('minimum_confidence', .8))
    maximum_abs = float(spec.get('maximum_abs', 1e6))
    offset_seconds = float(spec.get('offset_seconds', 0.))
    period_seconds = float(spec.get('period_seconds', .01))
    def data_lines():
        if not header:
            yield first
        for line in iterator:
            if line.strip():
                yield line
    def chunks():
        chunk = []
        for line in data_lines():
            chunk.append(line)
            if len(chunk) == _FEATURE_CHUNK_ROWS:
                yield chunk
                chunk = []
        if chunk:
            yield chunk
    for chunk in chunks():
        matrix = _feature_numeric_chunk(chunk, delimiter, len(first_row), selected,
                                       width if names is None else None, rows)
        timestamps = (matrix[:, -1] if time_index is not None else
                      offset_seconds + np.arange(rows, rows + len(chunk), dtype=np.float64) * period_seconds)
        bad_time = ~np.isfinite(timestamps) | (timestamps < 0)
        bad_time[1:] |= timestamps[1:] < timestamps[:-1]
        if last_time is not None:
            bad_time[0] |= timestamps[0] < last_time
        if bad_time.any():
            raise ValueError('invalid_or_nonmonotonic_feature_timestamp:' + str(rows + int(np.flatnonzero(bad_time)[0])))
        last_time = float(timestamps[-1])
        rows += len(chunk)
        confidence = matrix[:, quality_positions['confidence']] if 'confidence' in quality_positions else np.ones(len(chunk))
        success = matrix[:, quality_positions['success']] if 'success' in quality_positions else np.ones(len(chunk))
        good = np.isfinite(confidence) & (confidence >= minimum_confidence) & (success == 1.)
        values = matrix[:, :width]
        valid = np.isfinite(values) & (np.abs(values) <= maximum_abs) & good[:, None]
        safe = np.where(valid, values, 0.)
        squared = safe * safe
        for owner, union in enumerate(intervals):
            bounds = np.searchsorted(timestamps, union, side='left')
            for start, stop in bounds:
                if stop <= start:
                    continue
                count[owner] += valid[start:stop].sum(axis=0, dtype=np.int64)
                sums[owner] += safe[start:stop].sum(axis=0)
                squares[owner] += squared[start:stop].sum(axis=0)
    mean = np.divide(sums, count, out=np.zeros_like(sums), where=count > 0)
    variance = np.divide(squares, count, out=np.zeros_like(sums), where=count > 0) - mean * mean
    features = np.concatenate((mean, np.sqrt(np.maximum(variance, 0.))), axis=1).astype(np.float32)
    valid = np.concatenate((count > 0, count > 0), axis=1)
    return features, valid
