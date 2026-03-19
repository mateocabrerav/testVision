def process_tracking_data(x_percent, area_percent):
    """
    Process tracking data for the best matched object in the current frame.
    
    Args:
        x_percent (float): Horizontal position (-100% left to 100% right).
        area_percent (float): Bounding box area relative to frame size (0-100%).
    """
    # You can add your custom logic here
    # For now, we just print the values
    print(f"Tracking Data -> X: {x_percent:.1f}%, Area: {area_percent:.1f}%")
