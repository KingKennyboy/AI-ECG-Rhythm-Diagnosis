
def check_csv_columns(data, expected_cols=12):
    for row in data:
        if len(row) != expected_cols:
            return False
    return True


def is_csv(data):
    try:
        _ = next(data)
        return True
    except Exception:
        return False