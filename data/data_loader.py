
def CreateDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader


def CreatePredictDataLoader(opt):
    from data.custom_dataset_data_loader import CustomDatasetDataLoader, CustomDatasetDataLoader2
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    data_loader2 = CustomDatasetDataLoader2()
    print(data_loader2.name())
    data_loader2.initialize(opt)
    return data_loader, data_loader2

