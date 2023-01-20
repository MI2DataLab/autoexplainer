#tests how does the current verion of latex/pdf report work

from pylatex import Document, Section, Subsection, Figure, Tabu, Center, Command, Itemize
from pylatex.utils import italic, bold, NoEscape
import pandas as pd
import numpy as np
import os

def to_pdf(output_path, data, model_name, dataset_name, execution_time, best_result):

    tex_file=output_path
    pdf_file='development/report/full8.pdf'
    if os.path.exists(tex_file):
        os.remove(tex_file)
    if os.path.exists(pdf_file):
        os.remove(pdf_file)
    if os.path.exists(tex_file):
        print("exists")
    else:
        print("does not exist")
        
    pwd=os.getcwd()
    image_filename = f'{pwd}/src/lazy_explain/img/mini_logo.png'
    left_margin=2
    max_nr_columns_in_table=5
    geometry_options = {"tmargin": "1cm", "lmargin": f"{left_margin}cm"}
    doc = Document(geometry_options=geometry_options)
    doc.preamble.append(Command('title', 'Report'))
    doc.append(NoEscape(r'\maketitle'))

    with doc.create(Section('General information')):
        doc.append(bold('Model name: '))
        doc.append(italic(f'{model_name} \n'))
        doc.append(bold('Dataset name: '))
        doc.append(italic(f'{dataset_name} \n'))
        doc.append(bold('Execution time: '))
        doc.append(italic(f'{execution_time}'))

    with doc.create(Section('Table of results')):
        for j in range((len(data.columns)-1)//max_nr_columns_in_table+1):
            if j !=0:
                data_small = pd.concat([data.iloc[:,0], data.iloc[:, j*max_nr_columns_in_table:(j+1)*max_nr_columns_in_table]], axis=1)
            else:
                data_small = data.iloc[:, j*max_nr_columns_in_table:(j+1)*max_nr_columns_in_table]
            tabu_str=""
            for i in range(len(data_small.columns)):
                tabu_str += "X[r] "

            with doc.create(Center()) as centered:
                with centered.create(Tabu(tabu_str, to=f"{21-2*left_margin}cm")) as data_table:
                    header_row1 = data_small.columns
                    data_table.add_row(header_row1, mapper=[bold], color="blue")
                    data_table.add_empty_row()
                    data_table.add_hline()
                    for i in range(0,len(data_small)):
                        data_table.add_row(data_small.loc[i])
                        data_table.add_hline(color="lightgray")

    with doc.create(Section('Best result')):
        doc.append(bold('Best explanation method: '))
        doc.append(italic(f'{best_result} \n'))

    with doc.create(Section('Details')):
        with doc.create(Subsection('Explanation methods')):
            with doc.create(Itemize()) as itemize:
                for i in range(1, len(data.columns)):
                    itemize.add_item(bold(data.columns[i]))
                    doc.append(': short description')
                    doc.append('\n')
           
        with doc.create(Subsection('Metrics')):
            with doc.create(Itemize()) as itemize:
                for i in range(0, len(data)):
                    itemize.add_item(bold(data.iloc[i,0]))
                    doc.append(': short description')
                    doc.append('\n')

    with doc.create(Section('Authors')):
            doc.append('Lazy-explain project team')
            doc.append('\n')

    with doc.create(Figure(position='h!')) as mini_logo:
        mini_logo.add_image(image_filename, width='120px')    
    tex=doc.dumps()
    with open(output_path, 'w') as f:
        f.write(tex)     

    doc.generate_pdf('development/report/full8', clean_tex=False)


#testing
# with open('tests/report_tests/first_aggregation_results.pkl', 'rb') as f:
    # data = pickle.load(f)
data={'kernel_shap': {'faithfulness_estimate': 0.14080895447930578, 
'average_sensitivity': 0.48409177362918854, 'irof': 1.0998467502414613, 
'sparseness': 0.3573727472611998, 'FaithfulnessCorrelation':0.6, 'Complexity': 0.5, 'RandomLogit': 0.9}, 'integrated_gradients':
 {'faithfulness_estimate': 0.43731802874231823, 'average_sensitivity': 0.03979700896888971, 
 'irof': 1.0995387416931948, 'sparseness': 0.6869805698584132, 'FaithfulnessCorrelation':0.6, 'Complexity': 0.5, 'RandomLogit': 0.9}, 'grad_cam': 
 {'faithfulness_estimate': 0.6216781469604523, 'average_sensitivity': 0.16170302033424377, 
 'irof': 1.0927817816722118, 'sparseness': 0.39891754201452134, 'FaithfulnessCorrelation':0.6, 'Complexity': 0.5, 'RandomLogit': 0.9}, 'saliency': 
 {'faithfulness_estimate': 0.6462111254101452, 'average_sensitivity': 0.04281927831470966, 
 'irof': 1.0933935374725436, 'sparseness': 0.5017330314977393, 'FaithfulnessCorrelation':0.6, 'Complexity': 0.5, 'RandomLogit': 0.9}, 'lrp_captum': 
 {'faithfulness_estimate': 0.6462111254101452, 'average_sensitivity': 0.04281927831470966, 
 'irof': 1.0933935374725436, 'sparseness': 0.5017330314977393, 'FaithfulnessCorrelation':0.6, 'Complexity': 0.5, 'RandomLogit': 0.9}}
print(data)
dict_names={
                "faithfulness_estimate": "Faithfulness Estimate",
                "FaithfulnessCorrelation": "Faithfulness Correlation",
                "MaxSensitivity": "Max Sensitivity", 
                "Complexity": "Complexity", 
                "RandomLogit":"Random Logit",
                "irof":"Iterative Removal Of Features",
                "average_sensitivity":"Average Sensitivity",
                "sparseness":"Sparseness",

                "kernel_shap": "Kernel SHAP",
                "integrated_gradients": "Integrated Gradients",
                "grad_cam": "Grad CAM",
                "saliency": "Saliency",
                "lrp_captum":"LRP Captum",
                "custom_shap_quantus_interface": "Custom SHAP",
                "lrp_epsilon_plus_flat_zennit":"LRP Epsilon",
                "lime_quantus_interface":"LIME", 

                "explanation name":"explanation name"
                }
explanation_methods=list(data.keys())
metrics=list(data[explanation_methods[0]].keys())
explanation_methods_new=[]


for i in range(len(explanation_methods)):
    explanation_methods_new.append(dict_names[explanation_methods[i]])

print(explanation_methods)
print(explanation_methods_new)

metrics=["explanation name"]+metrics
df=np.zeros((len(explanation_methods),len(metrics)))
pd_df=pd.DataFrame(df,columns=metrics)
for i in range(len(explanation_methods)):
    pd_df.iloc[i,0]=explanation_methods_new[i]
    for j in range(1,len(metrics)):
        pd_df.iloc[i,j]=round(float(np.mean(data[explanation_methods[i]][metrics[j]])),2)
pd_df.columns=[dict_names[metrics[i]] for i in range (len(metrics))]
print(pd_df.columns)

to_pdf('development/report/full8.tex', pd_df, 'model naaaame', 'dataseeeet name', 'tiiime', 'best result')