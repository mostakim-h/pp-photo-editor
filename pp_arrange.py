def ppArrange(arrangement):
    """Pretty-print an arrangement of items in a grid format.

    Args:
        arrangement (list of list): A 2D list representing the arrangement.
    Returns:
        str: A formatted string representing the grid.
    """
    if not arrangement or not all(isinstance(row, list) for row in arrangement):
        return "Invalid arrangement"

    # Determine the maximum width of each column
    col_widths = []
    num_cols = max(len(row) for row in arrangement)
    for col in range(num_cols):
        max_width = max((len(str(row[col])) if col < len(row) else 0) for row in arrangement)
        col_widths.append(max_width)

    # Create the formatted string
    formatted_rows = []
    for row in arrangement:
        formatted_row = " | ".join(f"{str(item):<{col_widths[i]}}" if i < len(row) else " " * col_widths[i]
                                    for i, item in enumerate(row))
        formatted_rows.append(formatted_row)

    return "\n".join(formatted_rows)