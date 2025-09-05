#!/usr/bin/env python
from spacetimeengine import *

def main():
    # Retrieves a known solution from the Solutions() class.
    blackhole_solution = Solution().schwarzschild()
    blackhole_spacetime = SpaceTime(blackhole_solution)

    print("Metric tensor (dd):")
    pprint(blackhole_spacetime.metric_tensor_dd)
    print("\nMetric tensor (uu):")
    pprint(blackhole_spacetime.metric_tensor_uu)
    print("\nRicci tensor (dd):")
    pprint(blackhole_spacetime.ricci_tensor_dd)
    print("\nRicci scalar:")
    pprint(blackhole_spacetime.ricci_scalar)


    # Plot Ricci scalar curvature for Schwarzschild spacetime
    # Adjust x_range and y_range as needed for your coordinates
    blackhole_spacetime.plot_ricci_scalar_grid(x_range=(2, 200), y_range=(0, 180), x_index=1, y_index=2, num_points=10)

if __name__ == "__main__":
    main()
