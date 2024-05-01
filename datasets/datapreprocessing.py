import pandas as pd

dataset = pd.read_csv('original_dataset.csv')

columns_to_convert = ['Gender', 'How many hours in a day do you spend on your smartphones, laptops, etc?',
                      'Eyes that are sensitive to light?', 'Eyes that feel gritty (itchy and Scratchy) ?',
                      'Painful or Sore eyes?', 'Blurred vision?', 'Poor Vision?', 'Reading?',
                      'Driving at night?', 'Working with a computer or bank machine ATM?', 'Watching TV?',
                      'Windy conditions?', 'Places or areas with low humidity (very dry)?', 'Areas that are air-conditioned?',
                      'Results']  # Add the names of columns you want to convert

for column in columns_to_convert:
    mapping_dict = {text_value: index for index,
                    text_value in enumerate(dataset[column].unique())}
    dataset[column] = dataset[column].replace(mapping_dict)

columns_to_drop = ['Timestamp', 'Consent', 'Academic Year',
                   'What type of Digital display device do you use?', 'OSDI', 'Unnamed: 21',
                   'Unnamed: 22', 'Unnamed: 23', 'Unnamed: 24', 'Unnamed: 25',
                   'Unnamed: 26', 'Unnamed: 27']
dataset.drop(columns=columns_to_drop, axis=1, inplace=True)
dataset.to_csv('preprocessed_dataset.csv', index=False)
# Address the FutureWarning
pd.set_option('future.no_silent_downcasting', True)
