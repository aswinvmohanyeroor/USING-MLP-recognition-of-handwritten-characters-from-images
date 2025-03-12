# Dataset class
class CustomDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.data_info = pd.read_csv(csv_file)  # Reading CSV file into a pandas DataFrame
        self.root_dir = root_dir  # Setting the root directory for images
        self.transform = transform  # Assigning image transformation function

    def __len__(self):
        return len(self.data_info)  # Returning the length of the dataset

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.data_info.iloc[idx, 0])  # Constructing the image path
        image = Image.open(img_name)  # Opening the image using PIL
        label = self.data_info.iloc[idx, 1]  # Extracting the label from the DataFrame
        # Convert label to numeric label using the predefined label_map
        label = label_map[label]
        if self.transform:
            image = self.transform(image)  # Applying transformation to the image if specified
        return image, label