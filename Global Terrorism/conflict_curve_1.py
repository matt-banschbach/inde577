import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

acled = pd.read_csv("Asia-Pacific_2018-2024_Nov22.csv")
# print(acled.head())

myanmar_raw = acled[acled.country == "Myanmar"]
# print(f"Raw file has {len(myanmar_raw)} observations from Myanmar")

myanmar21 = myanmar_raw[myanmar_raw.year >= 2021].copy()
# print(len(myanmar21), " observations from 2021 onward")
# print("Columns: ", myanmar21.columns)

# Remove 'non-violent' action
event_drop = ['Protests', 'Strategic developments', 'Riots']
myanmar21 = myanmar21[~myanmar21.event_type.isin(event_drop)].copy()
# myanmar21 = myanmar21[myanmar21.event_type == 'Battles'].copy()

myanmar21 = myanmar21.copy()
myanmar21['event_date'] = pd.to_datetime(myanmar21['event_date'])
myanmar21['month'] = myanmar21['event_date'].dt.to_period('M')

# Group by month and event type, and count the number of events
monthly_event_counts = myanmar21.groupby(['month', 'event_type']).size().unstack(fill_value=0)

# Plot the stacked bar chart
plt.figure(figsize=(12, 8))
monthly_event_counts.plot(kind='bar', stacked=True, color=['blue', 'red', 'green'])
plt.xlabel('Month')
plt.ylabel('# of Events')
plt.title('Number of Events per Month by Event Type')
# plt.xticks(rotation=45)
plt.legend(title='Event Type')
plt.tight_layout()
plt.show()