"""
To reproduce reports from the thesis, run this script with parameters:

script_name: path to this script
dataset_name: for datasets in this repository, choose from: Imagenette, chest_xray, KP
number_of_images: to run script on
path_to_images: if other data than the one provided in this repository, fill in this parameter with path to dataset, otherwise put '.'
how_many_times: to run the selection for provided dataset
path_to_folder_where_to_put_all_outputs: path to output directory

example: python development/eden/run_experiment_script.py Imagenette 2 . 1 development/eden/output_from_experiments

"""

import torch
import torchvision
import sys
import pickle
import os
import shutil
from autoexplainer.autoexplainer import AutoExplainer
from matplotlib import pyplot as plt

def load_model_and_data(number_of_images=2, dataset_name="Imagenette", path_to_img=""):
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.Resize((256, 256)),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    if dataset_name == "Imagenette":
        model = torch.load("development/models/chosen_model_imagenette_pretrained_097.pth", map_location="cuda") #change to cuda
        dataset = torchvision.datasets.ImageFolder("development/data/imagenette/", transform=transform)
        labels={0: "tench", 1: "English springer", 2:"cassette player", 3:"chain saw", 4:"church", 5:"French horn", 6:"garbage truck", 7:"gas pump", 8:"golf ball", 
                9:"parachute"}
    elif dataset_name == "chest_xray":
        model = torch.load("development/models/chosen_model_CXR_not_pretrained_092.pth", map_location="cuda") #change to cuda
        dataset = torchvision.datasets.ImageFolder("development/data/cxr/", transform=transform)
        labels={0:"COVID", 1: "Non-COVID", 2: "Normal"}
    elif dataset_name == "KP":
        model = torch.load("development/models/chosen_model_KP_challenge1_not_pretrained_09.pth", map_location="cuda") #change to cuda
        dataset = torchvision.datasets.ImageFolder("development/data/kandinsky_1_challenge/", transform=transform)
        labels={0: "counterfactual", 1:"false", 2:"true"}
    elif dataset_name == "Imagenette_all":
        model = torch.load("development/models/chosen_model_imagenette_pretrained_097.pth", map_location="cuda") #change to cuda
        dataset = torchvision.datasets.ImageFolder(path_to_img, transform=transform)
        labels={0: "tench", 1: "English springer", 2:"cassette player", 3:"chain saw", 4:"church", 5:"French horn", 6:"garbage truck", 7:"gas pump", 8:"golf ball", 
                9:"parachute"}
    elif dataset_name == "chest_xray_all":
        model = torch.load("development/models/chosen_model_CXR_not_pretrained_092.pth", map_location="cuda") #change to cuda
        dataset = torchvision.datasets.ImageFolder(path_to_img, transform=transform)
        labels={0:"COVID", 1: "Non-COVID", 2: "Normal"}
    elif dataset_name == "KP_all":
        model = torch.load("development/models/chosen_model_KP_challenge1_not_pretrained_09.pth", map_location="cuda") #change to cuda
        dataset = torchvision.datasets.ImageFolder(path_to_img, transform=transform) 
        labels={0: "counterfactual", 1:"false", 2:"true"}
    else:
        print("wrong arguments")
        return
    # for reproducibility of choosing images
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
    g = torch.Generator()
    g.manual_seed(25)
    
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=number_of_images, worker_init_fn=seed_worker, generator=g, shuffle=True)
    x_batch, y_batch = next(iter(data_loader))

    return model, x_batch, y_batch, labels

def main(argv):
    input_arguments=argv
    dataset_name=input_arguments[1]
    number_of_images=int(input_arguments[2])
    how_many_times=int(input_arguments[4])
    dir_path_output=str(input_arguments[5])
    path_to_img=str(input_arguments[3])
    #dataset name in argv
    for i in range (how_many_times):
        model, x_batch, y_batch, labels = load_model_and_data(number_of_images=number_of_images, dataset_name=dataset_name, path_to_img=path_to_img)
        explainer = AutoExplainer(model, x_batch, y_batch, device="cuda")
        explainer.evaluate()
        explainer.aggregate()
        #mkdir for every run
        whole_path_to_dir=os.path.join(dir_path_output, f'output_{dataset_name}_num_img_{number_of_images}')
        if os.path.exists(whole_path_to_dir):
            shutil.rmtree(whole_path_to_dir)
        os.mkdir(whole_path_to_dir)
        explainer.to_html(f'{whole_path_to_dir}/example_report_{dataset_name}_{i}.html', model_name='DenseNet121', labels=labels)
        explainer.to_pdf(f'{whole_path_to_dir}', model_name='DenseNet121', dataset_name=dataset_name, labels=labels)

        best_explanation = explainer.get_best_explanation()
        new_attributions = best_explanation.explain(model, x_batch, y_batch)
        
        def save_to_pickle(path, obj):
            if os.path.exists(path):
                os.remove(path)
            with open(path, 'wb') as f:
                pickle.dump(obj, f)

        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_best_explanation_{i}_numOfImgsIs_{number_of_images}.pkl', best_explanation)
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_new_attributions_{i}_numOfImgsIs_{number_of_images}.pkl', new_attributions)
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_explainer_{i}_numOfImgsIs_{number_of_images}.pkl', explainer)
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_raw_results_{i}_numOfImgsIs_{number_of_images}.pkl', explainer.raw_results)
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_after_first_aggregation_{i}_numOfImgsIs_{number_of_images}.pkl', explainer.first_aggregation_results)

        ## times exported to 3 pickles:  divide by num of images to get avg time per image
        # save times of each method - attributions only 
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_times_methods_{i}_numOfImgsIs_{number_of_images}.pkl', explainer.times_methods)
        # save times of each metric with details for each method - including attributions above
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_times_metrics_{i}_numOfImgsIs_{number_of_images}.pkl', explainer.times_metrics)
        # save times of each metric, aggregated by metric
        save_to_pickle(f'{whole_path_to_dir}/{dataset_name}_times_metrics_aggregated_{i}_numOfImgsIs_{number_of_images}.pkl', explainer.times_metrics_aggregated)


        ##### times plot methods - start - #####
        times_methods = explainer.times_methods
        keys = list(times_methods.keys())
        mapped_keys = {
            'kernel_shap': "KernelSHAP",
            'integrated_gradients': "Integrated\nGradients",
            'grad_cam': "Grad-CAM",
            'saliency': "Saliency"
        }

        fig = plt.figure(figsize=(5, 3), dpi=300)
        ax = fig.add_subplot(111)
        ax.set_title("Time spent computing attributions for all images\nusing different explanation methods")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("Explanation method")
        ax.bar([mapped_keys[k] for k in keys], [times_methods[k]/number_of_images for k in keys], width=0.6)
        # add text labels
        for j, v in enumerate([times_methods[k] / number_of_images for k in keys]):
            ax.text(j - 0.2, v + 0.13, str(round(v, 2)), color='black')
        ax.set_ylim(0, max(times_methods.values()) * 1.085 / number_of_images)
        fig.savefig(f"{whole_path_to_dir}/{dataset_name}_times_methods_{i}_numOfImgsIs_{number_of_images}.png", bbox_inches='tight')
        ##### times plot methods -  end  - #####

        ##### times plot metrics - start - #####
        times_metrics_aggregated = explainer.times_metrics_aggregated
        keys = list(times_metrics_aggregated.keys())
        mapped_keys = {
            'faithfulness_estimate' : "Faithfulness\nEstimate",
            'average_sensitivity' : "Average\nSensitivity",
            'irof' : "IROF",
            'sparseness' : "Sparseness"
        }

        fig = plt.figure(figsize=(5, 3), dpi=300)
        ax = fig.add_subplot(111)
        ax.set_title("Time spent\nevaluating explanation methods for one image\nusing different metrics")
        ax.set_ylabel("Time (seconds)")
        ax.set_xlabel("Metric")
        ax.bar([mapped_keys[k] for k in keys], [times_metrics_aggregated[k] / number_of_images for k in keys], width=0.6)
        y_scale_max = max(times_metrics_aggregated.values()) * 1.085 / number_of_images
        # add text labels
        ax.set_ylim(0, y_scale_max)
        for j, v in enumerate([times_metrics_aggregated[k] / number_of_images for k in keys]):
            ax.text(j - 0.2, v + y_scale_max * 0.016, str(round(v, 2)), color='black')
        fig.savefig(f"{whole_path_to_dir}/{dataset_name}_times_metrics_aggregated_{i}_numOfImgsIs_{number_of_images}.png", bbox_inches='tight')
        ##### times plot metrics -  end  - #####

if __name__=="__main__":
    main(sys.argv)



