class Dataset:
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def get_item(self, index):
        return self.data[index]
    
    def __get_item__(self, index):
        return self.data[index]

d = Dataset([1, 2, 3, 4, 5])
print(d.get_item(1))
print(d.__get_item__(2))
print(len(d))
    