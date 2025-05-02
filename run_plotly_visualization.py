#!/usr/bin/env python3
"""
Script to run the Plotly-based visualization of the HJ reachability level set.
"""

from run.racecar_warm_hj_plotly import RacecarWarmHJPlotlyRunner

def main():
    """
    Main function to run the simulation with Plotly visualization.
    """
    # Create the runner
    runner = RacecarWarmHJPlotlyRunner()
    
    # Run the simulation
    runner.run()

if __name__ == "__main__":
    main()