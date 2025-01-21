# inspired by Chip Huyen's https://github.com/chiphuyen/aie-book/blob/main/scripts/ai-heatmap.ipynb 
from datetime import datetime, timedelta
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import numpy as np

class GhostHeatmap:
    def __init__(self, year=2025):
        self.year = year
        self.ghost = self.create_patterns()
        
    def create_patterns(self):
        """Create ghost pattern using numeric matrix for different intensities"""
        # Ghost pattern (properly oriented)
        ghost = [
            [0,0,1,1,1,0,0],    # Top
            [0,1,1,1,1,1,0],    # Upper body
            [1,0,0,1,0,0,1],    # Face
            [1,2,0,1,2,0,1],    # Eyes
            [1,1,1,1,1,1,1],    # Lower body
            [1,1,1,1,1,1,1],    # Bottom
            [1,0,1,1,0,1,1]     # Feet
        ]
        return ghost 
    
    def get_first_monday(self):
        """Get the first Monday of the year"""
        date = datetime(self.year, 1, 1)
        while date.weekday() != 0:  # 0 represents Monday
            date += timedelta(days=1)
        return date
    
    def generate_activity_data(self, weeks_offset=1):
        """Generate calendar activity data with ghost patterns"""
        activities = {}
        
        # Initialize calendar with background
        start_date = datetime(self.year, 1, 1)
        end_date = datetime(self.year, 12, 31)
        current_date = start_date
        
        while current_date <= end_date:
            # Create varied background noise
            rand_val = random.random()
            if rand_val < 0.2:  # 40% chance of no activity
                activities[current_date.strftime('%Y-%m-%d')] = 0
            elif rand_val < 0.7:  # 30% chance of very light activity
                activities[current_date.strftime('%Y-%m-%d')] = 1
            elif rand_val < 0.85:  # 15% chance of light activity
                activities[current_date.strftime('%Y-%m-%d')] = 2
            else:  # 15% chance of medium-light activity
                activities[current_date.strftime('%Y-%m-%d')] = 3
            current_date += timedelta(days=1)
        
        # Get the first Monday and calculate starting position
        first_monday = self.get_first_monday()
        pattern_start = first_monday + timedelta(weeks=weeks_offset)
        
        # Place 6 ghosts with spacing
        base_spacing = 8  # Pattern width (7) + 1 week space
        for ghost_num in range(6):
            ghost_offset = ghost_num * base_spacing
            ghost_start = pattern_start + timedelta(weeks=ghost_offset)
            
            for week in range(len(self.ghost[0])):
                for day in range(len(self.ghost)):
                    if self.ghost[day][week] > 0:  # Check if it's not white (0)
                        date = ghost_start + timedelta(weeks=week, days=day)
                        if date.year == self.year:
                            date_str = date.strftime('%Y-%m-%d')
                            # Set intensity based on pattern value: 1 for green, 2 for dark green
                            if self.ghost[day][week] == 1:
                                activities[date_str] = 10  # Darker regular green
                            else:
                                activities[date_str] = 48  # Darker dark green
        
        return activities
    
    # most credit to Chip Huyen's code https://github.com/chiphuyen/aie-book/blob/main/scripts/ai-heatmap.ipynb
    def create_heatmap(self, data, output_file="ghost_heatmap.png"):
        """Create calendar heatmap visualization"""
        # Calculate date range
        start_date = datetime(self.year, 1, 1)
        end_date = datetime(self.year, 12, 31)
        total_days = (end_date - start_date).days + 1
        weeks_in_year = total_days // 7 + 1

        # Setup plot
        plt.figure(figsize=(15, 8))
        ax = plt.gca()
        ax.set_aspect('equal')

        # Calculate statistics for color scaling
        values = list(data.values())
        p90_value = np.percentile(values, 90)
        max_day = max(data.items(), key=lambda x: x[1])
        max_count_date = max_day[0]
        max_count = max_day[1]

        # Plot calendar grid
        current_date = start_date
        while current_date <= end_date:
            # Calculate week number relative to year start
            week = ((current_date - start_date).days + start_date.weekday()) // 7
            day_num = current_date.weekday()
            
            date_str = current_date.strftime('%Y-%m-%d')
            activity = data.get(date_str, 0)
            
            # Use the same color scaling as reference
            color = plt.cm.Greens((activity + 1) / p90_value) if activity > 0 else 'lightgray'
            
            # Draw calendar cell
            rect = patches.Rectangle(
                (week, day_num),
                1, 1,
                linewidth=0.5,
                edgecolor='black',
                facecolor=color
            )
            ax.add_patch(rect)
            
            current_date += timedelta(days=1)

        # Add month labels
        current_date = start_date
        while current_date <= end_date:
            if current_date.day == 1:
                week = ((current_date - start_date).days + start_date.weekday()) // 7
                plt.text(week + 0.5, 7.75, current_date.strftime('%b'), 
                        ha='center', va='center', fontsize=10, rotation=0)
            current_date += timedelta(days=1)

        # Customize plot appearance
        ax.set_xlim(-0.5, weeks_in_year + 0.5)
        ax.set_ylim(-0.5, 8.5)
        
        # Add title with stats
        plt.title(
            f'{self.year} ChatGPT Conversation Heatmap (total={sum(data.values())}).\n'
            f'Most active day: {max_count_date} with {max_count} convos.',
            fontsize=16
        )
        
        # Set axis labels
        plt.xticks([])
        plt.yticks(range(7), ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
        plt.gca().invert_yaxis()
        
        # Save and close
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Create heatmap for year 2025
    generator = GhostHeatmap(2025)
    activity_data = generator.generate_activity_data(weeks_offset=1)
    
    # Save activity data
    with open('ghost_activity.json', 'w') as f:
        json.dump(activity_data, f, indent=2)
    
    # Generate heatmap
    generator.create_heatmap(activity_data)
    print("Generated ghost heatmap!")

if __name__ == "__main__":
    main()