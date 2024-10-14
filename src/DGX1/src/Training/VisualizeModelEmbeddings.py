from tqdm import tqdm
import random

def visualize_model_embeddings(model, dataset, device,name=None,number_of_samples=None):
    """
    Visualize the model embeddings
    :param model: The model
    :param dataset: The dataset
    :param device: The device
    """
    model.eval()
    model.to(device)
    embeddings = []
    labels = []
    if number_of_samples:
        samples = random.sample(range(len(dataset)), number_of_samples)
    else:
        samples = range(len(dataset))
        number_of_samples = len(dataset)



    for idx in tqdm(samples, desc="Visualizing embeddings",total=number_of_samples):
        image, label = dataset.get_image_location(idx)

        # add batch dimension
        image = image.unsqueeze(0)


        device_image = image.to(device)
        output = model(device_image)
        output = output.cpu().detach().numpy()
        embeddings.append(output)
        labels.append(label)
    

    with open(f"{name if name else ''}_vectors.tsv", "w") as f:
        for element in embeddings:
            f.write("\t".join([str(x) for x in element[0][0]])+"\n")

    with open(f"{name if name else ''}_metadata.tsv", "w") as f:
        for label in labels:
            f.write(label+"\n")