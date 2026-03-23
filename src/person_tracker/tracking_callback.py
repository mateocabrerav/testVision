def process_tracking_data(x_percent: float, area_percent: float, distance_m: float = 0.0):
    """
    Process tracking data for the best matched object in the current frame.
    
    Args:
        x_percent: Horizontal position (-100% left to 100% right).
        area_percent: Bounding box area relative to frame size (0-100%).
        distance_m: Estimated distance to target in metres.
    """
    print(f"Tracking Data -> X: {x_percent:.1f}%, Area: {area_percent:.1f}%, Dist: {distance_m:.1f}m")
