import pydoc
import matplotlib
import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
from copy import deepcopy

import os
import pickle
import time
import random
import colorsys


import utils
import cpils

import warnings
warnings.filterwarnings("ignore")

from scipy.spatial.distance import cdist
from itertools import combinations
from tqdm.notebook import tqdm
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from typing import Any
from typing import List
from sklearn.base import BaseEstimator

class MultiClass_CP_ILS():
   
    """
    A multi-class extension of the CP_ILS method designed for binary classification.
    
    This class handles multi-class classification scenarios by transforming them into binary classification problems in different modes. 
    The supported modes include 1v1, 1vAll, 1vAll_Spec, Allv1, and All_Specv1, which specifies how the multi-class problem is decomposed into binary ones.


    Constants:
    ----------
    
    MODE_1V1 : str
        Constant representing the 1-vs-1 mode.
    MODE_1VALL : str
        Constant representing the 1-vs-All mode.
    MODE_1VALL_SPEC : str
        Constant representing the 1-vs-All (Specific) mode.
    MODE_ALLV1 : str
        Constant representing the All-vs-1 mode.
    MODE_ALL_SPECV1 : str
        Constant representing the All-vs-1 (Specific) mode.

    Attributes:
    ----------
    
    flatten : function
        A lambda function to flatten multi-dimensional arrays.
    __mbbox_model : model
        The multi-class black-box classifier model used for predictions, also referred as MBB
    __train_set : pd.DataFrame
        The training dataset used to generate latent spaces.
    __test_set : pd.DataFrame
        The test dataset used for validation and counterfactual generation.
    __mode : str
        The mode of classification, which defines how the multi-class problem is treated.
    __class_combinations : list
        A list of class combinations generated according to the mode.
        
    __dict_train_datasets : dict
        A dictionary storing the training datasets for each class combination.
    __dict_test_datasets : dict
        A dictionary storing the test datasets for each class combination.
    __dict_latent_spaces : dict
        A dictionary storing the latent spaces for each class combination.
    __hex_colors_for_classes : dict
        A dictionary storing the hexadecimal colors for each class, used for plotting points with same class color in all exp.

    Public Methods:
    --------
    
    __init__(self, bbox_model, train_set, test_set, target_class="class", mode=MODE_1V1):
        Initializes the MultiClass_CP_ILS object with the given dataset, model, and classification mode.
        
    generate_ls(self, latent_dim_k=2, early_stop_param=5, timer=False):
        Generates latent spaces for each class combination using CP-ILS method, leveraging a black-box model.

    generate_counterfactuals(self, test_instance, change_f, max_f, counter_classes=-1, filtering_mode="", debug=True):
        Generates counterfactual explanations for a given test instance by modifying the input features for a specific mode

    execute_ranking_of_counterfactuals(self, test_instance, df_counterfactuals, ranking="proximity", group_by_class=True, top_k=-1):
        Function to rank with a certain ranking criterion a list of counterfactuals, obtained respect to a test_instance

    plot_ls_comb(self,target,comb=None,class_name=None):
        Function to plot a specific combination of latent space callable from external.
    
    plot_ls_comb_compare(self,comb=None,class_name=None):
       Function to plot and compare specific combination of latent space callable from external.
    
    plot_ls_training_losses(self,comb=None,class_name=None):
        Function to plot specific combination of train_test losses callable from external.

    execute_testing_on_data(self,test_name,test_instances,change_f,max_f,filtering_mode="acceptable",debug=False):
        Automatic procedure to print on external files the counterfactuals main information and treat them separately

    save_latent_spaces(self, filename):
        Saves the latent spaces to a file using pickle.

    load_latent_spaces(self, filename):
        Loads latent spaces from a file using pickle.

    get_ls_of_class(self, class_name):
        Returns the latent spaces specific to a given class.

    get_ls_of_not_class(self, class_name):
        Returns the latent spaces for all classes except the specified one.
        
        
    Private Methods:
    --------
    
    __adjust_target_class(self, target_name, train_set,test_set):
        Adjusts the target class column to be the last column in the dataset and renames it to 'class'.

    __build_dataset_idx_num_cat(self):
        Builds index lists for numerical and categorical attributes in the dataset.

    __assign_color_to_classes(self):
        Assigns a unique color to each class using HLS color space.

    __prepare_datasets(self):
        Prepares datasets for the specified classification mode by splitting them into appropriate combinations of classes.
    __generate_counterfactuals_1v1(self,test_instance,change_f,max_f,y_pred,counter_classes=-1,filtering_mode="",debug=True):
        Generates counterfactual explanations for a given test instance using the modality 1v1 

    __generate_counterfactuals_1vAll_AllSpec(self,test_instance,change_f,max_f,y_pred,counter_classes=-1,filtering_mode="",debug=True):
        Generates counterfactual explanations for a given test instance using the modality 1vALL or 1vALLSpec

    __generate_counterfactuals_Allv1_Allv1Spec(self,test_instance,change_f,max_f,y_pred,counter_classes=-1,filtering_mode="",debug=True):
        Generates counterfactual explanations for a given test instance using the modality Allv1 or AllSpecV1

    __plot_ls_directions(self,latent,X_instances,title=""):
        Function to plot the binary classifier counterfactuals direction in a latent space

    __plot_ls_class_scatter_plot(self,Z_train,Y_labels,title="",target="classes_MBB",comb=("None","All")):
        Function to plot the 2D scatter plot of a latent space, optionally targeting a specific classes combination 

    __prepare_ls_class_scatter_plot(self,ax,Z_train,Y_labels,title="",comb=("None","All")):
        Function to plot the 2D scatter plot of a latent space comparing the MBB with the BBB

    __plot_ls_losses(self,losses,title=""):
        Function to plot the losses of training and testing during the latent spaces generations

    """

    #Constants
    MODE_1V1="1v1"
    MODE_1VALL="1vAll"
    MODE_1VALL_SPEC="1vAll_Spec"
    MODE_ALLV1="Allv1"
    MODE_ALL_SPECV1="All_Specv1"
    
    def __init__(self,bbox_model: BaseEstimator,train_set: pd.DataFrame,test_set: pd.DataFrame,bbox_structure:BaseEstimator,target_class:str="class",mode:str=MODE_1V1):
        """
            Constructor method for initialization of attributes, preparing datasets in proper format and for class combinations.
            Setting the target class with name class if different, locate what indexes are categorical and numerical, build the
            needed combination used for generation of latent spaces and further to generate counterfactuals depending on the mode
            passed.
            
            Args:
                bbox_model (BaseEstimator): A scikit-learn trained model used for multi-class classification (black box model).
                train_set (pd.DataFrame): DataFrame containing the training dataset.
                test_set (pd.DataFrame): DataFrame containing the test dataset.
                target_class (str): The target class to predict. Default is "class".
                bbox_structure (BaseEstimator): The black box model to train inside the specific combination
                mode (str): The mode for generating counterfactuals. Default is MODE_1V1.
        """
        
        MODE_ALLOWED=[self.MODE_1V1,self.MODE_1VALL,self.MODE_1VALL_SPEC,self.MODE_ALLV1,self.MODE_ALL_SPECV1]
        
        #Global utility function for flatteing, used also in original CP_ILS
        self.flatten = lambda m: [item for row in m for item in row]

        #Preliminary: check if target and mode passed are correct 
        if (target_class not in train_set.columns.values):
            print(f"{target_class} is not a valid target class. \nPlease insert one of {train_set.columns.values}")
            return

        if (mode not in MODE_ALLOWED):
            print(f" {mode} is not one of the allowed modality. Please insert one of  {MODE_ALLOWED}")
            return

        self.__adjust_target_class(target_class,train_set,test_set)
    
        #Global attributes
        self.__mbbox_model=bbox_model
        self.__train_set=train_set.astype(np.float64)
        self.__test_set=test_set.astype(np.float64)
        self.__mode=mode
        self.__bbox_structure=deepcopy(bbox_structure)
 
        #Attributes related to the dataset provided
        self.__build_dataset_idx_num_cat()
         
        #Group of dataset per combination of classes
        self.__class_combinations=[]
        self.__dict_train_datasets={}
        self.__dict_test_datasets={}

        #Latent spaces
        self.__dict_latent_spaces={}
        
        self.__prepare_datasets() 

    ## Utility TO BE REMOVED (just for quick init)
    """ def get_ls(self):
        return self.__dict_latent_spaces """
        
    ## Utility TO BE REMOVED (just for quick init)
    """ def set_ls(self,latent_spaces):
        self.__dict_latent_spaces=latent_spaces """

    
    def save_latent_spaces(self, filename: str):
        """
            Save the generated latent spaces into a file.

            Args:
                filename (str): The path where the latent spaces will be saved.
        """
        with open(filename, "wb") as file:
            pickle.dump(self.__dict_latent_spaces, file)
        print(f"Latent space saved in {filename}")

    def load_latent_spaces(self,filename:str):
        """
            Load the generated latent spaces from a file.
            
            Args:
                filename (str): The string containing the path to load the file
        """
        with open(filename, "rb") as file:
            self.__dict_latent_spaces=pickle.load(file)
            print(f"Latent space loaded from {filename}")

    def get_ls_of_class(self, class_name:str): 
        """
            Method to return the latent space where the class with class_name is involved.
            
            Args:
                class_name (str): The name of the class to be reserched in latents spaces combinations
            
            Returns:
                    list: A list of tuples, where each tuple contains a latent space key and its corresponding value,
                          filtered to include only those where class_name is present in the key.
        """
        return [(key,value) for key,value in zip(self.__dict_latent_spaces.keys(),self.__dict_latent_spaces.values()) if class_name in key]

    def get_ls_of_not_class(self, class_name: str):
        """
        Method to return the latent space where the class with class_name is NOT involved.

        Args:
            class_name (str): The name of the class to be researched in latent space combinations.

        Returns:
            list: A list of tuples, where each tuple contains a latent space key and its corresponding value,
                filtered to include only those where class_name is NOT present in the key.
        """
        return [(key,value) for key,value in zip(self.__dict_latent_spaces.keys(),self.__dict_latent_spaces.values()) if class_name not in key]


    ## Utility TO BE REMOVED (just for test)
    """ def get_dict_train_dataset(self):
        return self.__dict_train_datasets """

    ## Utility TO BE REMOVED (just for test)
    """ def get_dict_test_dataset(self):
        return self.__dict_test_datasets """
    
    ## Utility TO BE REMOVED (just for test)    
    """ def get_hex_colors(self):
        return self.__hex_colors_for_classes """

    
##########     PRE-PROCESSING PART    ##########

    def __adjust_target_class(self,target_name:str,train_set:pd.DataFrame,test_set:pd.DataFrame):
        """
            This function move at the last column position the target class and rename it, if necessary, as class.
            In addition, for each possible class values, it define a color saved in data structure, used to plot the 2D latent spaces scatter plot.
            
            Args:
                target_name (str): The class name to be found in the train_set
                train_set (pd.DataFrame): One of the set where the target class should be moved at last column position
                test_set (pd.DataFrame): One of the set where the target class should be moved at last column position
        """
        
        #The class column must be fixed at last position
        new_position = -1 
        cols = train_set.columns.tolist()
        
        # Remove the column to move and reinsert it at the new position
        cols.insert(new_position, cols.pop(cols.index(target_name)))
        
        train_set = train_set[cols]
        train_set.rename(columns={target_name: 'class'}, inplace=True)

        #Do the same for test set
        new_position = -1 
        cols = test_set.columns.tolist()
        cols.insert(new_position, cols.pop(cols.index(target_name)))
        test_set = test_set[cols]
        test_set.rename(columns={target_name: 'class'}, inplace=True)

        classes=train_set["class"].unique()
        classes.sort()
        
        self.__all_classes=classes
   
        print(f"Target class: {target_name}" + (f"\nNow renamed class" if target_name != "class" else ""))
        print(f"Class values are: {self.__all_classes}")
        self.__assign_color_to_classes()
        
        
    def __build_dataset_idx_num_cat(self):

        """
            This function aims to populate the variable __idx_num_cat as in original work of CP_ILS.
            The __idx_num_cat is a variable containg the dataset information about continous and categorical attributes.
            The categorical attribute are one-hot-encoded. This variable is necessary to correctly handles categorical columns during 
            the generation of latent space and counterfactuals.

            It is a list of list, where each sub list contains the indexes referred to the same attribute. 
            E.g if we have 2 Attributes, Age and Gender, where gender can assume the 2 categorical values(M-F) after one hot encoding:
            the total columns are 3: Age, Gender_M, Gender_F and the list is composed by the elements [[0],[1,2]].

        """
        idx_num_cat=[] 

        #Fix as starting column_name the name of the class that for sure it is not present
        column_name=self.__train_set.columns[-1]
        
        #Detection of one hot categorical feature, that contains _ and consider as unique variable
        for index, column in enumerate(self.__train_set.columns[:-1]):
        
            if (not column.startswith(column_name)):
                new_list=[]
                idx_num_cat.append(new_list)
                column_name=column.split("_")[0]
            
            new_list.append(index)
            
        self.__idx_num_cat=idx_num_cat
        
        #Save separately the numerical and categorical indexes 
        self.idx_num = self.flatten([l for l in self.__idx_num_cat if len(l)==1])
        self.idx_cat = self.flatten([l for l in self.__idx_num_cat if len(l)>1])

        print(f"Numerical indexes: {self.idx_num}")
        print(f"Categorical indexes: {self.idx_cat}")
        print(f"{self.__idx_num_cat}")

    def __assign_color_to_classes(self):
        """
            This function assign significantly different colors for each class values saving it as dict
            self.__hex_colors_for_classes data structure.
        """
        #Choose a set of color fixed for classes
        hues = [i / len(self.__all_classes) for i in range(len(self.__all_classes))]  # Distribute hues evenly
        saturation, lightness = 0.65, 0.5  # Fixed saturation and lightness for balance
        colors = {cls: colorsys.hls_to_rgb(hue, lightness, saturation) for cls, hue in zip(self.__all_classes, hues)}
        hex_colors = {cls: to_hex(color) for cls, color in colors.items()}
        self.__hex_colors_for_classes=hex_colors


    def __prepare_datasets(self):
        """
            This function aims to preparing the training and testing dataset for each possible class combinations depending on the
            mode selected at the begin.
            In 1v1 mode: for each possible pair of classes "combinations" filter the samples that are involved for that binary classification
            In 1vAll or Allv1 mode: for each possible class select its samples entierly and then selecting randomly the remaining (n_classes-1) in equal proportion
            to obtain the same number of samples of the target class.
            In 1vAllSpec or Allv1Spec: the procedure is the same of 1vAll-Allv1 but not chosen the sample randomly but the ones most near to the target class.

        """

        #Calling this method, at this point the training and testing dataset are supposed to be balanced.
        
        print(f"\nModality:  {self.__mode}")
        
        if (self.__mode==self.MODE_1V1):
            
            for comb in combinations(self.__all_classes,2):
                self.__class_combinations.append(comb)
    
                class_i=comb[0] #e.g 1
                class_j=comb[1] #e.g 3
                
                print(f"Preparation of datasets for combination {comb}")
                self.__dict_train_datasets[comb]=pd.concat([self.__train_set[self.__train_set["class"]==class_i],self.__train_set[self.__train_set["class"]==class_j]]).astype(np.float64)
                self.__dict_test_datasets[comb]=pd.concat([self.__test_set[self.__test_set["class"]==class_i],self.__test_set[self.__test_set["class"]==class_j]]).astype(np.float64)

        if (self.__mode==self.MODE_1VALL or self.__mode==self.MODE_ALLV1):
            
            for class_value in self.__all_classes:
                comb=(class_value,"All")
                self.__class_combinations.append(comb)

                class_i=comb[0] #e.g 1
                class_j=comb[1] #e.g ALL

                print(f"Preparation of datasets for combination {comb}")

                #Get the number of instances of the target class and obtain the number of sample to use for the other classes
                train_class_i=self.__train_set[self.__train_set["class"]==class_i]
                no_train_samples_per_class=int(train_class_i.shape[0]/(len(self.__all_classes)-1))

                test_class_i=self.__test_set[self.__test_set["class"]==class_i]
                no_test_samples_per_class=int(test_class_i.shape[0]/(len(self.__all_classes)-1))

                train_set_copy=self.__train_set.copy()
                test_set_copy=self.__test_set.copy()

                #Create new dataframes sampling accordingly to the number obtained, in order to have a balanced datasets.
                final_train_set=pd.DataFrame()
                final_test_set=pd.DataFrame()
                
                for cls_value in self.__all_classes:
                    if (cls_value!=class_i):
                        no_eff_min_train_samples=min(train_set_copy[train_set_copy["class"]==cls_value].shape[0],no_train_samples_per_class)
                        no_eff_min_test_samples=min(test_set_copy[test_set_copy["class"]==cls_value].shape[0],no_test_samples_per_class)
                        final_train_set=pd.concat([final_train_set, train_set_copy[train_set_copy["class"]==cls_value].sample(n=no_eff_min_train_samples,random_state=42)]).reset_index(drop=True)
                        final_test_set=pd.concat([final_test_set, test_set_copy[test_set_copy["class"]==cls_value].sample(n=no_eff_min_test_samples,random_state=42)]).reset_index(drop=True)
                    else:
                        final_train_set=pd.concat([final_train_set, train_set_copy[train_set_copy["class"]==cls_value].sample(n=no_train_samples_per_class*(len(self.__all_classes)-1),random_state=42)]).reset_index(drop=True)
                        final_test_set=pd.concat([final_test_set, test_set_copy[test_set_copy["class"]==cls_value].sample(n=no_test_samples_per_class*(len(self.__all_classes)-1),random_state=42)]).reset_index(drop=True)

                #**CONVENTION**:
                #Lonley class "1" is encoded with class 0
                #Grouped class "All" is encoded with class 1
                #This is useful for the testing part in the generation of counterfactual to determine if they are acceptable or not
                
                #Binarization of the dataset impose as class=0 the class_i, the first in the combination and 1 the others
                final_train_set.loc[final_train_set["class"]==class_i,"class"]=-1
                final_train_set.loc[(final_train_set["class"]!=class_i) & (final_train_set["class"]!=-1),"class"]=1
                final_train_set.loc[(final_train_set["class"]==-1),"class"]=0
                
                #Binarization of the dataset impose as class=0 the class_i, the first in the combination and 1 the others
                final_test_set.loc[final_test_set["class"]==class_i,"class"]=-1
                final_test_set.loc[(final_test_set["class"]!=class_i) & (final_test_set["class"]!=-1),"class"]=1
                final_test_set.loc[(final_test_set["class"]==-1),"class"]=0
 
                self.__dict_train_datasets[comb]=final_train_set
                self.__dict_test_datasets[comb]=final_test_set

        if (self.__mode==self.MODE_1VALL_SPEC or self.__mode==self.MODE_ALL_SPECV1):

            for class_value in self.__all_classes:
                comb=(class_value,"All_Spec")
                self.__class_combinations.append(comb)
    
                class_i=comb[0] #e.g 1
                class_j=comb[1] #e.g ALL
    
                print(f"Preparation of datasets for combination {comb}")

                #Get the number of instances of the target class and obtain the number of sample to use for the other classes
                train_class_i=self.__train_set[self.__train_set["class"]==class_i]
                no_train_samples_per_class=int(train_class_i.shape[0]/(len(self.__all_classes)-1))
                
                test_class_i=self.__test_set[self.__test_set["class"]==class_i]
                no_test_samples_per_class=int(test_class_i.shape[0]/(len(self.__all_classes)-1))

                final_train_set=pd.DataFrame()
                final_train_set=pd.concat([final_train_set,train_class_i.sample(n=no_train_samples_per_class*(len(self.__all_classes)-1),random_state=42)],axis=0).reset_index(drop=True)
                
                #For each remaining dataset, the points are chosen using the shortest proximity distance to the class_i cluster in even proportion
                for class_value_rim in [x for x in self.__all_classes if x != class_value]:
                    
                    #Compute a matrix of distances among multidimentional points and selecting the lowest values
                    train_class_j=self.__train_set[self.__train_set["class"]==class_value_rim]
                    dists=self.cdist_sample_distance(train_class_i,train_class_j)
                    flat_indices = np.argsort(dists, axis=None)
                    row_indices, col_indices = np.unravel_index(flat_indices, dists.shape)
                    
                    unique_col_indices = []
                    for c in col_indices:
                        if c not in unique_col_indices:
                            unique_col_indices.append(c)
                        if len(unique_col_indices) == no_train_samples_per_class:
                            break
                    final_train_set=pd.concat([final_train_set,train_class_j.iloc[unique_col_indices,:]],axis=0).reset_index(drop=True)

                #The test dataset instead is chosen randomly because it is not possibile to know in advance what data could be presented
                test_set_copy=self.__test_set.copy()
                final_test_set=pd.DataFrame()
                
                for cls_value in self.__all_classes:
                    if (cls_value!=class_i):
                        final_test_set=pd.concat([final_test_set, test_set_copy[test_set_copy["class"]==cls_value].sample(n=no_test_samples_per_class,random_state=42)]).reset_index(drop=True)
                    else:
                        final_test_set=pd.concat([final_test_set, test_set_copy[test_set_copy["class"]==cls_value].sample(n=no_test_samples_per_class*(len(self.__all_classes)-1),random_state=42)]).reset_index(drop=True)


                #**CONVENTION**:
                #Lonley class "1" is encoded with class 0
                #Grouped class "All" is encoded with class 1
                #This is useful for the testing part in the generation of counterfactual to determine if they are acceptable or not
                
                #Binarization of the dataset impose as class=0 the class_i, the first in the combination and 1 the others
                final_train_set.loc[final_train_set["class"]==class_i,"class"]=-1
                final_train_set.loc[(final_train_set["class"]!=class_i) & (final_train_set["class"]!=-1),"class"]=1
                final_train_set.loc[(final_train_set["class"]==-1),"class"]=0

                #Binarization of the dataset impose as class=0 the class_i, the first in the combination and 1 the others
                final_test_set.loc[final_test_set["class"]==class_i,"class"]=-1
                final_test_set.loc[(final_test_set["class"]!=class_i) & (final_test_set["class"]!=-1),"class"]=1
                final_test_set.loc[(final_test_set["class"]==-1),"class"]=0
                
                self.__dict_train_datasets[comb]=final_train_set
                self.__dict_test_datasets[comb]=final_test_set
                
                

##########     OPERATIVE PART    ##########
    def generate_ls(self,latent_dim_k:int=2,early_stop_param:int=5,timer:bool=False):
        """
            This method compute the latent spaces for all the combinations of classes according to the initial selected modality
            and using as training and testing dataset the specific prepared for each combination.
            For each combination, a new black box classifier is trained with the sub-selected training samples, then the CP_ILS
            classical method is called in order to build this specific binary latent space.
            
            The results of binary black box, latent space and training values are saved into self.__dict_latent_spaces dict variable.

            Args:
                latent_dim_k (int): the dimension K of each generated latent space by default it is equal to 2
                early_stop_param (int): the condition to avoid to generate too many iterations in case of no improvments, defulat 5 epochs.
                timer (bool): boolean variable to take note of the time elapsed for training
        """

        if timer:
            start_time = time.time()
        
        for comb in self.__class_combinations:
            print(f"Training latent space for combination: {comb}")
            
            class_i=comb[0]
            class_j=comb[1]

            label_encoder = LabelEncoder()
            sub_ij_training_dataset=self.__dict_train_datasets[comb].copy()
            sub_ij_test_dataset=self.__dict_test_datasets[comb].copy()
 
            sub_ij_training_dataset["class"]=label_encoder.fit_transform(sub_ij_training_dataset["class"].astype(np.int64))
            sub_ij_test_dataset["class"]=label_encoder.transform(sub_ij_test_dataset["class"].astype(np.int64)) 

            binary_bb = deepcopy(self.__bbox_structure)
            binary_bb.fit(sub_ij_training_dataset.iloc[:,:-1],sub_ij_training_dataset.iloc[:,-1])
            
            latent = cpils.CP_ILS(binary_bb.predict, binary_bb.predict_proba, latent_dim=latent_dim_k, early_stopping=early_stop_param)
            losses = latent.fit((sub_ij_training_dataset.iloc[:,:-1].values, sub_ij_test_dataset.iloc[:,:-1].values), self.__idx_num_cat, seed=42)
            
            self.__dict_latent_spaces[comb]={"binary_bbox":binary_bb, "label_encoder": label_encoder, "latent":latent, "losses": losses}

        if timer:
            end_time = time.time() 
            print(f"Time needed to train all the latent spaces {(end_time-start_time):.2f} sec")

    def generate_counterfactuals(self,test_instance:pd.DataFrame,change_f:List[int],max_f:int,counter_classes:Any=-1,filtering_mode:str="",debug:bool=True):
        """
            Main function to generate counterfactuals for a specific instance. This function should be called after having trained all the latent spaces.
            Depending on the modality initially chosen, it applies the default generate counterfacutal method on each combination.

            Args:
                test_instance (pd.DataFrame): the instance interested in counterfactuals explanations
                change_f (List[int]): the list of feature that can be modifiable during the procedure of generation
                max_f (int): it is the maximum number of attribute that can be changed togheter from the change_f list
                counter_classes (Any): it specify a list of possible classes values interested in generation of counterfactuals or -1 if all available
                filtering_mode (str): (optionally) it indicate the modality of possible filtering of counterfactuals 'acceptable','specific'
                debug (bool): boolean value to print on console debug informations
            
            Returns:
                list: a list of counterfactuals grouped by class, cotaining information about all the generated, the predictions of the multi-class and binary classifiers
                and the filtered ones.

        """
        if (debug): print(f"Modality: {self.__mode}")
        
        if (self.__mode==self.MODE_1V1 or self.__mode==self.MODE_ALLV1 or self.__mode==self.MODE_ALL_SPECV1):
            
            filtering_acceptable="'acceptable' considers any counterfactuals belonging to classes different from the predicted outcome of the Multi-classification black-box model"
            filtering_specific="'specific' considers only counterfactuals belonging to a specific class, different from the predicted outcome of the Multi-classification black-box model"
            
            if filtering_mode!="acceptable" and filtering_mode!="specific":
                print(f"filtering_mode can be only 'acceptable' or 'specific'.\n{filtering_acceptable}\n{filtering_specific}")
                return

            if (debug): print(f"Starting generation of counterfactuals using filtering_mode: {filtering_mode} \n"f"{filtering_acceptable if filtering_mode == 'acceptable' else filtering_specific}")
        
        else:

            if filtering_mode!="":
                print(f"In this mode {self.__mode} you cannot specify any filtering method. All the counterfactuals are acceptable.")
                return

            if (debug): print(f"Starting generation of counterfactuals ")

        
        statistics={}
        y_pred=self.__mbbox_model.predict(test_instance)
        if (debug): print(f"\nThe predicted class from MBB is {y_pred}\n")

        if (counter_classes!=-1 and y_pred in counter_classes):
            print("The counter class you want is the same as the predicted class")
            return

        
        if (self.__mode==self.MODE_1V1):
            return self.__generate_counterfactuals_1v1(test_instance,change_f,max_f,y_pred,counter_classes,filtering_mode,debug)

        if (self.__mode==self.MODE_1VALL or self.__mode==self.MODE_1VALL_SPEC):
            return self.__generate_counterfactuals_1vAll_AllSpec(test_instance,change_f,max_f,y_pred,counter_classes,filtering_mode,debug)

        if (self.__mode==self.MODE_ALLV1 or self.__mode==self.MODE_ALL_SPECV1):
            return self.__generate_counterfactuals_Allv1_Allv1Spec(test_instance,change_f,max_f,y_pred,counter_classes,filtering_mode,debug)
            
        return None

    def __generate_counterfactuals_1v1(self,test_instance:pd.DataFrame,change_f:List[int],max_f:int,y_pred:int,counter_classes:Any=-1,filtering_mode:str="",debug:bool=True):

        """
            This function should be called after having trained all the latent spaces.
            It applies the default generate counterfacutal method on each combination and take the counterfacutals that are 
            acceptable and optionally filtering the specifics.

            Args:
                test_instance (pd.DataFrame): the instance interested in counterfactuals explanations
                change_f (List[int]): the list of feature that can be modifiable during the procedure of generation
                max_f (int): it is the maximum number of attribute that can be changed togheter from the change_f list
                y_pred (int): the predicted class of the multi-class classifier
                counter_classes (Any): it specify a list of possible classes values interested in generation of counterfactuals or -1 if all available
                filtering_mode (str): (optionally) it indicate the modality of possible filtering of counterfactuals 'acceptable','specific', default 'acceptable'
                debug (bool): boolean value to print on console debug informations
            
            Returns:
                list: a list of counterfactuals grouped by class, cotaining information about all the generated, the predictions of the multi-class and binary classifiers
                and the filtered ones.
        """

        #Looking in all latent spaces where the class predicted is involved in order to follow the direct direction,
        # from 1 the specific class to (1, an other specific class). 
        list_of_latent_spaces_to_be_considered=self.get_ls_of_class(y_pred)
        
        statistics={}
        for elem in list_of_latent_spaces_to_be_considered:
            comb=elem[0] # here is expected to find a tuple like (1,3)
            latent=elem[1]["latent"]
            
            class_of_counterfactual= comb[1] if comb[0]==y_pred else comb[0]
            
            if (counter_classes==-1 or class_of_counterfactual in counter_classes):

                statistics[class_of_counterfactual]={}
                if (debug): print(f"Generation of counterfactual class {class_of_counterfactual}")

                counterfactuals=latent.get_counterfactuals(test_instance.astype(np.float64), change_f, max_f, max_steps=50, n_cfs=-1, n_feats_sampled=10, topn_to_check=10, seed=42).reset_index(drop=True)  

                if (not counterfactuals.empty):
                    
                    bbox=elem[1]["binary_bbox"]
                    
                    y_counter_prediction=bbox.predict(counterfactuals)   
                    label_encoder=elem[1]["label_encoder"]
                    if (debug): print(f"BBB Encoded y_pred: {y_counter_prediction}")
                    
                    y_counter_prediction_decoded=label_encoder.inverse_transform(y_counter_prediction)
                    if (debug): print(f"BBB Decoded y_pred: {y_counter_prediction_decoded}")

                    y_counter_prediction_mbbox=self.__mbbox_model.predict(counterfactuals).copy()
                    
                    #Save all the generated counterfactuals and their predictions indipendently they are correct or not
                    statistics[class_of_counterfactual]["all_counterfactuals"]=counterfactuals.copy()
                    statistics[class_of_counterfactual]["bbbox_predictions"]=y_counter_prediction_decoded.copy()
                    statistics[class_of_counterfactual]["mbbox_predictions"]=y_counter_prediction_mbbox.copy()

                    #Discard the countefactuals that fails under binary-classifier: 
                    #this means the methods it self has failed in discovering an acceptable counterfactual
                    indexes_with_diff_label=np.where(y_counter_prediction_decoded!=class_of_counterfactual)
                    counterfactuals.drop(counterfactuals.index[indexes_with_diff_label[0]], inplace=True) 

                    tot_counter_produced_binary=len(counterfactuals)
                    
                    y_counter_prediction_mbbox=np.delete(y_counter_prediction_mbbox, indexes_with_diff_label)
                    if (debug): print(f"MBB Original y_pred: {y_counter_prediction_mbbox}")

                    #Discard the counterfactuals that fails under multi-class classifier: the class is the same as starting class
                    indexes_with_same_starting_label=np.where(y_counter_prediction_mbbox==y_pred)
                    counterfactuals.drop(counterfactuals.index[indexes_with_same_starting_label[0]], inplace=True) 
                    y_counter_prediction_mbbox=np.delete(y_counter_prediction_mbbox, indexes_with_same_starting_label)

                    tot_counter_acceptable=len(counterfactuals)
                    if (debug): print(f"acceptable counterfactuals: {tot_counter_acceptable}")
                    
                    #Discard all the counterfactuals different from the ORIGINAL class_of_counterfactual designed
                    if (filtering_mode=="specific"): 
                        indexes_with_diff_label=np.where(y_counter_prediction_mbbox!=class_of_counterfactual)
                        counterfactuals.drop(counterfactuals.index[indexes_with_diff_label[0]], inplace=True)
                        tot_counter_correct=len(counterfactuals) 
                        if (tot_counter_correct!=tot_counter_acceptable):
                            if (debug): print(f"Discarded counterfactuals: {tot_counter_acceptable-tot_counter_correct}\n")
                    
                    statistics[class_of_counterfactual]["counterfactuals"]=counterfactuals
                else:
                    statistics[class_of_counterfactual]["all_counterfactuals"]=None
                    statistics[class_of_counterfactual]["counterfactuals"]=None
                    
        return statistics


    def __generate_counterfactuals_1vAll_AllSpec(self,test_instance:pd.DataFrame,change_f:List[int],max_f:int,y_pred:int,counter_classes:Any=-1,filtering_mode:str="",debug:bool=True):

        """
            This function should be called after having trained all the latent spaces.
            It applies the default generate counterfacutal method on a specific combination where the y_pred is involved (directly) in
            the sense of the class y_pred is on the 1 side, instead the others are on the ALL side.
            This method takes only the counterfacutals that are acceptable.

            Args:
                test_instance (pd.DataFrame): the instance interested in counterfactuals explanations
                change_f (List[int]): the list of feature that can be modifiable during the procedure of generation
                max_f (int): it is the maximum number of attribute that can be changed togheter from the change_f list
                y_pred (int): the predicted class of the multi-class classifier
                counter_classes (Any): it specify a list of possible classes values interested in generation of counterfactuals or -1 if all available
                filtering_mode: (str): should be "" value because only acceptable counterfactuals are allowed
                debug (bool): boolean value to print on console debug informations
            
            Returns:
                list: a list of counterfactuals grouped by class, cotaining information about all the generated, the predictions of the multi-class and binary classifiers
                and the filtered ones.
        """

        #Looking in all latent spaces where the class predicted is involved (just 1) in order to follow the direct direction,
        # from 1 the specific class to All, the opposite classes labels of the predicted one. 
        
        list_of_latent_spaces_to_be_considered=self.get_ls_of_class(y_pred)
        
        statistics={}
        for elem in list_of_latent_spaces_to_be_considered:
            comb=elem[0] # here is expected to find a tuple like (1,All)
            latent=elem[1]["latent"]
            
            #**CONVENTION** --> remember init the class of "1" side has value 0
            class_of_counterfactual=1
            
            statistics["All"]={}
            if (debug): print(f"Generation of counterfactual class ALL")
            
            counterfactuals=latent.get_counterfactuals(test_instance.astype(np.float64), change_f, max_f, max_steps=50, n_cfs=-1, n_feats_sampled=10, topn_to_check=10, seed=42).reset_index(drop=True)  
            
            if (not counterfactuals.empty):
                    
                bbox=elem[1]["binary_bbox"]
                        
                y_counter_prediction=bbox.predict(counterfactuals)   
                label_encoder=elem[1]["label_encoder"]
                
                y_counter_prediction_decoded=label_encoder.inverse_transform(y_counter_prediction)
                if (debug): print(f"BBB Decoded y_pred: {y_counter_prediction_decoded}")
                if (debug): print(f"BBB 1 means binary correctly classified")
                y_counter_prediction_mbbox=self.__mbbox_model.predict(counterfactuals).copy()

                #Save all the generated counterfactuals and their predictions indipendently they are correct or not
                statistics["All"]["all_counterfactuals"]=counterfactuals.copy()
                statistics["All"]["bbbox_predictions"]=y_counter_prediction_decoded.copy()
                statistics["All"]["mbbox_predictions"]=y_counter_prediction_mbbox.copy()

                #Discard the countefactuals that fails under binary classifier
                indexes_with_diff_label=np.where(y_counter_prediction_decoded!=class_of_counterfactual)

                counterfactuals.drop(counterfactuals.index[indexes_with_diff_label[0]], inplace=True) 
                
                tot_counter_produced_binary=len(counterfactuals)
                
                y_counter_prediction_mbbox=np.delete(y_counter_prediction_mbbox, indexes_with_diff_label)
                if (debug): print(f"MBB Original y_pred: {y_counter_prediction_mbbox}")

                #Discard the counterfactuals that fails under multi-class classifier: the class is the same as starting class
                indexes_with_same_starting_label=np.where(y_counter_prediction_mbbox==y_pred)
                counterfactuals.drop(counterfactuals.index[indexes_with_same_starting_label[0]], inplace=True) 
                y_counter_prediction_mbbox=np.delete(y_counter_prediction_mbbox, indexes_with_same_starting_label)

                tot_counter_acceptable=len(counterfactuals)
                if (debug): print(f"acceptable counterfactuals: {tot_counter_acceptable}")

                #If counter_classes is specified filter only the classes indicated
                if (counter_classes!=-1):
                    if (debug): print(f"Filtering counterfactuals of classes: {counter_classes}")
                    indexes_with_diff_label=np.where(~np.isin(y_counter_prediction_mbbox, counter_classes))
                    counterfactuals.drop(counterfactuals.index[indexes_with_diff_label[0]], inplace=True)
                    tot_counter_correct=len(counterfactuals)
                    if (tot_counter_correct!=tot_counter_acceptable):
                        if (debug): print(f"Discarded counterfactuals: {tot_counter_acceptable-tot_counter_correct}\n")
                statistics["All"]["counterfactuals"]=counterfactuals
                
            else:
                statistics["All"]["all_counterfactuals"]=None
                statistics["All"]["counterfactuals"]=None
                
        return statistics

    def __generate_counterfactuals_Allv1_Allv1Spec(self,test_instance:pd.DataFrame,change_f:List[int],max_f:int,y_pred:int,counter_classes:Any=-1,filtering_mode:str="",debug:bool=True):

        """
            This function should be called after having trained all the latent spaces.
            It applies the default generate counterfacutal method on each combination and take the counterfacutals that are 
            acceptable and optionally filtering the specifics. 
            It applies the default generate counterfacutal method on all the combinations where the y_pred is involved (indirectly) in
            the sense of the class y_pred is on the ALL side, instead the others are on the 1 side.
            

            Args:
                test_instance (pd.DataFrame): the instance interested in counterfactuals explanations
                change_f (List[int]): the list of feature that can be modifiable during the procedure of generation
                max_f (int): it is the maximum number of attribute that can be changed togheter from the change_f list
                y_pred (int): the predicted class of the multi-class classifier
                counter_classes (Any): it specify a list of possible classes values interested in generation of counterfactuals
                filtering_mode (str): (optionally) it indicate the modality of possible filtering of counterfactuals 'acceptable','specific',default 'acceptable'
                debug (bool): boolean value to print on console debug informations
            
            Returns:
                list: a list of counterfactuals grouped by class, cotaining information about all the generated, the predictions of the multi-class and binary classifiers
                and the filtered ones.
        """


        #Looking in all latent spaces where the class predicted is involved in order to follow the opposite direction,
        #from ALL (where y_pred also belong to 1 a specific_class who the counterfactuals is desidered.
    
        list_of_latent_spaces_to_be_considered=self.get_ls_of_not_class(y_pred)

        statistics={}
        for elem in list_of_latent_spaces_to_be_considered:
            comb=elem[0] # class_vs_all
            comb_first_elem=comb[0] #e.g. 1
            latent=elem[1]["latent"]

            #**CONVENTION** --> remember init the class of "1" side has value 0
            class_of_counterfactual=0

            if (counter_classes==-1 or comb_first_elem in counter_classes):

                statistics[comb_first_elem]={}
                if (debug): print(f"Generation of counterfactual class {comb_first_elem} in {comb}")

                counterfactuals=latent.get_counterfactuals(test_instance.astype(np.float64), change_f, max_f, max_steps=50, n_cfs=-1, n_feats_sampled=10, topn_to_check=10, seed=42).reset_index(drop=True)  
    
                if (not counterfactuals.empty):
    
                    bbox=elem[1]["binary_bbox"]
                    y_counter_prediction=bbox.predict(counterfactuals)   
                    label_encoder=elem[1]["label_encoder"]
                    
                    y_counter_prediction_decoded=label_encoder.inverse_transform(y_counter_prediction)
                    if (debug): print(f"BBB Decoded y_pred: {y_counter_prediction_decoded}")
                    
                    y_counter_prediction_mbbox=self.__mbbox_model.predict(counterfactuals).copy()
                    
                    #Save all the generated counterfactuals and their predictions indipendently they are correct or not
                    statistics[comb_first_elem]["all_counterfactuals"]=counterfactuals.copy()
                    statistics[comb_first_elem]["bbbox_predictions"]=y_counter_prediction_decoded.copy()
                    statistics[comb_first_elem]["mbbox_predictions"]=y_counter_prediction_mbbox.copy()
        
                    #Discard the countefactuals that fails under binary classifier
                    indexes_with_diff_label=np.where(y_counter_prediction_decoded!=class_of_counterfactual)
                    counterfactuals.drop(counterfactuals.index[indexes_with_diff_label[0]], inplace=True) 
        
                    tot_counter_produced_binary=len(counterfactuals)
        
                    y_counter_prediction_mbbox=np.delete(y_counter_prediction_mbbox, indexes_with_diff_label)
                    if (debug): print(f"MBB Original y_pred: {y_counter_prediction_mbbox}")
        
                    #Discard the counterfactuals that fails under multi-class classifier: the class is the same as starting class
                    indexes_with_same_starting_label=np.where(y_counter_prediction_mbbox==y_pred)
                    counterfactuals.drop(counterfactuals.index[indexes_with_same_starting_label[0]], inplace=True) 
                    y_counter_prediction_mbbox=np.delete(y_counter_prediction_mbbox, indexes_with_same_starting_label)
        
                    tot_counter_acceptable=len(counterfactuals)
                    if (debug): print(f"acceptable counterfactuals: {tot_counter_acceptable}")
        
                    #Discard all the counterfactuals different from the ORIGINAL class_of_counterfactual designed
                    if (filtering_mode=="specific"): 
                        indexes_with_diff_label=np.where(y_counter_prediction_mbbox!=comb_first_elem)
                        counterfactuals.drop(counterfactuals.index[indexes_with_diff_label[0]], inplace=True)
                        tot_counter_correct=len(counterfactuals) 
                        if (tot_counter_correct!=tot_counter_acceptable):
                            if (debug): print(f"Discarded counterfactuals: {tot_counter_acceptable-tot_counter_correct}\n")
                                
        
                    statistics[comb_first_elem]["counterfactuals"]=counterfactuals
                else:
                    statistics[comb_first_elem]["all_counterfactuals"]=None
                    statistics[comb_first_elem]["counterfactuals"]=None
        return statistics

    
    
    def execute_ranking_of_counterfactuals(self, test_instance:pd.DataFrame, df_counterfactuals:pd.DataFrame, ranking:str="proximity", group_by_class:bool=True, top_k:int=-1):
        """
            This function is used for ranking a list of counterfactuals related to a target test instance.
            It gives the possibility to rank counterfactuals per proximity or per probability class prediction.
            
            Args:
                test_instance (pd.DataFrame): the instance related to the counterfactuals list, used to determine proximity
                df_counterfactuals (pd.DataFrame): the list of counterfactuals that needs to be ranked
                ranking (str): the modality of ranking; proximity-> use the distance between the counters and the test_instance, 
                         accuracy pushes up the counterfactuals with higher prediction probability, default proximity
                group_by_class (bool): specifies if the results are presented ranked per class or not, default True
                top_k (int): specifies the number of counterfactulas to show at maximum in the final view
                
            Returns:
                list: It return a list of ranked counterfactuals accordingly to specified criterion.

        """
        results={}

        #Determine the probability prediction and proximity distance
        probab=self.__mbbox_model.predict_proba(df_counterfactuals)
        classes=np.argmax(probab,axis=1)
        max_prob=np.max(probab, axis=1)
        distances=self.cdist_sample_distance(test_instance,df_counterfactuals)
        
        #print (probab)
        #print (classes)
        #print (max_prob)
        
        df_counterfactuals["class"]=classes
        df_counterfactuals["prob"]=max_prob
        df_counterfactuals["dist"]=distances[0]

        
        if (ranking=="accuracy"):
            sorted_counterfactuals=df_counterfactuals.sort_values("prob", ascending=False)
        if (ranking=="proximity"):
            sorted_counterfactuals=df_counterfactuals.sort_values("dist", ascending=True)

        #If group the results by class or not
        if (group_by_class):
            counter_classes=sorted_counterfactuals["class"].unique()
            for counter_classes in counter_classes:
                ris={}
                select_counterfactuals=sorted_counterfactuals[sorted_counterfactuals["class"]==counter_classes]
                
                #Take the top_k ordered instances per class
                ris["counterfactuals"]=select_counterfactuals.head(top_k) if (top_k!=-1) else select_counterfactuals
                ris["prob"]=ris["counterfactuals"]["prob"].values
                ris["dist"]=ris["counterfactuals"]["dist"].values
                ris["counterfactuals"].drop(columns=["class","prob","dist"],inplace=True)
                ris["counterfactuals"]=ris["counterfactuals"].reset_index(drop=True)
                results[counter_classes]=ris
        else:
            ris={}
            #Take the top_k ordered instances among all
            ris["counterfactuals"]=sorted_counterfactuals.head(top_k) if (top_k!=-1) else sorted_counterfactuals
            ris["class"]=ris["counterfactuals"]["class"].values
            ris["prob"]=ris["counterfactuals"]["prob"].values
            ris["dist"]=ris["counterfactuals"]["dist"].values
            
            ris["counterfactuals"].drop(columns=["class","prob","dist"],inplace=True)
            ris["counterfactuals"]=ris["counterfactuals"].reset_index(drop=True)
            results=ris 
            
        return results


    def cdist_sample_distance(self, XA:pd.DataFrame, XB:pd.DataFrame, metric=('euclidean', 'jaccard'), w:Any=None):
        """
            The distance function used to measure the distance among to 2 distinct samples; the same used in original paper
            combining Hamming distance for categorical values and cosine distance for continous attribute.
            This method can be applied simultaneously to multiple points at time, returing a matrix of distances.

            Args:
                XA (pd.DataFrame): representing the first sample instance, can be used with multiple rows
                XB (pd.DataFrame): representing the second sample instance can be used with multiple rows
                metric (str): this is the criterion used for continuous and categorical variables
                w (Any): The weight vector for metrics that support weights (e.g., Minkowski)

            Returns:
                numpy.array: a numerical array of distances accordingly to metric between XA and XB 
        """
        metric_continuous = metric[0]
        metric_categorical = metric[1]

        if self.idx_cat:
            dist_categorical = cdist(XA.iloc[:, self.idx_cat], XB.iloc[:, self.idx_cat],
                                 metric=metric_categorical, w=w)
            #self.__train_set.shape[1]-1 = is the input_dim-1 of original dataset
            ratio_categorical = len(self.idx_cat) / (self.__train_set.shape[1]-1)
            dist = ratio_categorical * dist_categorical

            if self.idx_num:
                dist_continuous = cdist(XA.iloc[:, self.idx_num], XB.iloc[:, self.idx_num],
                                    metric=metric_continuous, w=w)
                #self.__train_set.shape[1]-1 = is the input_dim-1 of original dataset
                ratio_continuous = len(self.idx_num) / (self.__train_set.shape[1]-1)
                dist += ratio_continuous * dist_continuous 
        else:
            dist = cdist(XA, XB, metric=metric_continuous, w=w)

        return dist


##########     PLOTTING PART    ##########


    def __plot_ls_directions(self,latent:Any,X_instances: pd.DataFrame,title:str=""):
        """
            Function to plot the latent space counterfactuals directions
            
            Args:
                latent (Any): the latent space involved
                X_instances (pd.DataFrame): the original instances to be transformed and then projected in 2D space
                title (str): the title of the plot
        """

        plt.figure()
        W_train, Z_train = latent.transform(X_instances.iloc[:,:-1].values)

        if (Z_train.shape[1]!=2):
            print("The dimension of that latent space is greater then 2. The graph can not be plotted.")
            return
        plt.scatter(Z_train[:,0], Z_train[:,1], c=latent.y_train_bb, cmap='coolwarm')
        plt.title(title)
    
        for idx in np.random.choice(Z_train.shape[0], 100, replace=False):
            w = W_train.copy()
            y_contrib = w[:,-1]/np.linalg.norm(w[:,-1])
            if latent.y_train_bb[idx]>0.5:
                y_contrib *=-1
            plt.quiver(Z_train[idx,0], Z_train[idx,1],
                        y_contrib[0], y_contrib[1], angles='xy', scale_units='xy', width=5*1e-3, scale=20)
        plt.grid()

    def __plot_ls_class_scatter_plot(self,Z_train:pd.DataFrame,Y_labels:np.ndarray,title:str="",target:str="classes_MBB",comb:tuple=("None","All")):
        """
            Function to plot a scatter plot of specific points
            
            Args:
                Z_train (pd.DataFrame): the latent space samples involved
                Y_labels (np.ndarray): the original class values associated
                title (str): the title of the plot
                target (str): if it is involved the multi-class classifier or not classes_MBB (by default)
                comb (tuple): the specific combination involved in the plot
        """
        
        plt.figure()

        if (Z_train.shape[1]!=2):
            print("The dimension of that latent space is greater then 2. The graph can not be plotted.")
            return
            
        for label in np.unique(Y_labels):
            class_name=label
            if (self.__mode!=self.MODE_1V1 and target!="classes_MBB"):
                if (label==0):
                    class_name=comb[0]
                else:
                    class_name=comb[1]
            plt.scatter(Z_train[Y_labels==label,0], Z_train[Y_labels==label,1], c=self.__hex_colors_for_classes[label],label=f"Class {class_name}")
        
        plt.title(title)
        plt.legend(title="Classes")
    
        plt.grid()

    def __prepare_ls_class_scatter_plot(self,ax:matplotlib.axes.Axes,Z_train:pd.DataFrame,Y_labels:np.ndarray,title:str="",comb:tuple=("None","All")):
        """
            Function to produce a scatter plot to be used later in specific graph.
            
            Args:
                ax (matplotlib.axes.Axes): the variable involved in the plot
                Z_train (pd.DataFrame): the latent space samples involved
                Y_labels (np.ndarray): the original class values associated
                title (str): the title of the plot
                comb (tuple): the specific combination involved in the plot
            
            Returns:
                ax (matplotlib.axes.Axes): the updated variable with scatter plot
        """

        if (Z_train.shape[1]!=2):
            print("The dimension of that latent space is greater then 2. The graph can not be plotted.")
            return

        for label in np.unique(Y_labels):
            class_name=label
            label_color=label
            if (self.__mode!=self.MODE_1V1 and comb[0]!="None"):
                if (label==0):
                    class_name=comb[0]
                    label_color=comb[0]
                else:
                    class_name=comb[1]
                    label_color=value = next(key for key, val in self.__hex_colors_for_classes.items() if key != comb[0])
            ax.scatter(Z_train[Y_labels==label,0], Z_train[Y_labels==label,1], c=self.__hex_colors_for_classes[label_color],label=f"Class {class_name}")
        
        ax.set_title(title)
        ax.legend(title="Classes")
        ax.grid()
        
        return ax
                            
    def __plot_ls_losses(self,losses:list,title:str=""):
        """
            Function to plot the curves of training and validation losses during generation of latent spaces.
            
            Args:
                losses (list): the list of points to be visualized
                title (str): the title of the plot
            Displays:
                A plot showing the training and validation loss curves.
        """
        
        plt.figure()
        plt.title(title)
        plt.plot(losses[0], label='train')
        plt.plot(losses[1], label='test')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.yscale('log')
        plt.legend()
            
    def plot_ls_comb(self,target:str,comb:tuple=None,class_name:str=None):
        """
            
            Function callable from outside to plot a scatter plot of specific combination or class.
            
            Args:
                target (str): must specify the type of plot only possibles directions,classes_BBB or classes_MBB
                comb (tuple): the specific combination involved in the plot
                class_name (str): the specific class involved in the combinatio to be plotted
        """
        
        target_allowed=["directions","classes_BBB","classes_MBB"]
        if (target not in target_allowed):
            print(f"Target selected {target} is not allowed, please select 'directions' or 'classes'")
            return

        if (comb!=None and comb not in self.__class_combinations):
            print(f"{comb} is not a valid combination. Please select one from {self.__class_combinations}")
            return

        if (class_name!=None and class_name not in self.__all_classes):
            print(f"{class_name} is not a valid class name. Please select one from {self.__all_classes}")
            return
            
        if (target=="directions"):
            
            if (comb==None and class_name==None):
                for latent_space_comb in self.__dict_latent_spaces:
                    title = f"Plot of latent space of combination {latent_space_comb}"
                    sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb]
                    self.__plot_ls_directions(self.__dict_latent_spaces[latent_space_comb]["latent"],sub_ij_training_dataset,title)   
            
            if (comb!=None and comb in self.__class_combinations):
                sub_ij_training_dataset=self.__dict_train_datasets[comb]
                
                title = f"Plot of latent space of combination {comb}"
                self.__plot_ls_directions(self.__dict_latent_spaces[comb]["latent"],sub_ij_training_dataset,title) 
                return   
                    
            if (class_name!=None and class_name in self.__all_classes):
                for latent_space_comb in self.get_ls_of_class(class_name):
                    
                    title=f"Plot of latent space of combination {latent_space_comb[0]}"
                    sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb[0]]
                    self.__plot_ls_directions(latent_space_comb[1]["latent"],sub_ij_training_dataset,title)
                

        if (target=="classes_BBB" or target=="classes_MBB"):
            
            if (comb==None and class_name==None):
                for latent_space_comb in self.__dict_latent_spaces:

                    title = f"Plot of latent space of combination {latent_space_comb}"
                    sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb]

                    _, Z_train = self.__dict_latent_spaces[latent_space_comb]["latent"].transform(sub_ij_training_dataset.iloc[:,:-1].values)
                    
                    if (target=="classes_MBB"):
                        title=title+" -- MBB Multi-classification Black Box"
                        y_values=self.__mbbox_model.predict(sub_ij_training_dataset.iloc[:,:-1])
                    else:
                        title=title+" -- BBB Binary-classification Black Box"
                        y_values=self.__dict_latent_spaces[latent_space_comb]["binary_bbox"].predict(sub_ij_training_dataset.iloc[:,:-1])
                        y_values=self.__dict_latent_spaces[latent_space_comb]["label_encoder"].inverse_transform(y_values)
                    
                    self.__plot_ls_class_scatter_plot(Z_train,y_values.astype(np.int32),title,target,latent_space_comb)   
            
            if (comb!=None and comb in self.__class_combinations):

                title = f"Plot of latent space of combination {comb}"
                
                sub_ij_training_dataset=self.__dict_train_datasets[comb]
                _, Z_train = self.__dict_latent_spaces[comb]["latent"].transform(sub_ij_training_dataset.iloc[:,:-1].values)

                if (target=="classes_MBB"):
                        title=title+" -- MBB Multi-classification Black Box"
                        y_values=self.__mbbox_model.predict(sub_ij_training_dataset.iloc[:,:-1])
                else:
                        title=title+" -- BBB Binary-classification Black Box"
                        y_values=self.__dict_latent_spaces[comb]["binary_bbox"].predict(sub_ij_training_dataset.iloc[:,:-1])
                        y_values=self.__dict_latent_spaces[comb]["label_encoder"].inverse_transform(y_values)
                
                self.__plot_ls_class_scatter_plot(Z_train,y_values.astype(np.int32),title,target,comb)   
                return
                    
            if (class_name!=None and class_name in self.__all_classes):
                
                for latent_space_comb in self.get_ls_of_class(class_name):
                    
                    title=f"Plot of latent space of combination {latent_space_comb[0]}"
                    sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb[0]]

                    _, Z_train = self.__dict_latent_spaces[latent_space_comb[0]]["latent"].transform(sub_ij_training_dataset.iloc[:,:-1].values)
                    if (target=="classes_MBB"):
                        title=title+" -- MBB Multi-classification Black Box"
                        y_values=self.__mbbox_model.predict(sub_ij_training_dataset.iloc[:,:-1])
                    else:
                        title=title+" -- BBB Binary-classification Black Box"
                        y_values=self.__dict_latent_spaces[latent_space_comb[0]]["binary_bbox"].predict(sub_ij_training_dataset.iloc[:,:-1])
                        y_values=self.__dict_latent_spaces[latent_space_comb[0]]["label_encoder"].inverse_transform(y_values)
                    
                    self.__plot_ls_class_scatter_plot(Z_train,y_values.astype(np.int32),title,target,latent_space_comb[0])    
    
    def plot_ls_comb_compare(self,comb:tuple=None,class_name:str=None):

        """
            Function callable from outside to plot a comparison of scatter plots (multi-class and binary) for specific combination or class.
            
            Args:
                comb (tuple): the specific combination involved in the plot
                class_name (str): the specific class involved in the combinatio to be plotted
        """
        
        if (comb!=None and comb not in self.__class_combinations):
            print(f"{comb} is not a valid combination. Please select one from {self.__class_combinations}")
            return

        if (class_name!=None and class_name not in self.__all_classes):
            print(f"{class_name} is not a valid class name. Please select one from {self.__all_classes}")
            return


        if (comb==None and class_name==None):
            for latent_space_comb in self.__dict_latent_spaces:
                
                # Create two subplots, side by side (1 row, 2 columns)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
                title = ""
                sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb]
    
                _, Z_train = self.__dict_latent_spaces[latent_space_comb]["latent"].transform(sub_ij_training_dataset.iloc[:,:-1].values)
    
                title="MBB Multi-classification Black Box"
                y_values=self.__mbbox_model.predict(sub_ij_training_dataset.iloc[:,:-1])
                
                ax1=self.__prepare_ls_class_scatter_plot(ax1,Z_train,y_values,title)
    
                title="BBB Binary-classification Black Box"
                y_values=self.__dict_latent_spaces[latent_space_comb]["binary_bbox"].predict(sub_ij_training_dataset.iloc[:,:-1])
                y_values=self.__dict_latent_spaces[latent_space_comb]["label_encoder"].inverse_transform(y_values)
                
                ax2=self.__prepare_ls_class_scatter_plot(ax2,Z_train,y_values,title,latent_space_comb)
                
                fig.suptitle(f"Plot of latent space of combination {latent_space_comb}", fontsize=16)
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()
        
        if (comb!=None and comb in self.__class_combinations):

            # Create two subplots, side by side (1 row, 2 columns)
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                 
            sub_ij_training_dataset=self.__dict_train_datasets[comb]
            _, Z_train = self.__dict_latent_spaces[comb]["latent"].transform(sub_ij_training_dataset.iloc[:,:-1].values)

            title="MBB Multi-classification Black Box"
            y_values=self.__mbbox_model.predict(sub_ij_training_dataset.iloc[:,:-1])
            
            ax1=self.__prepare_ls_class_scatter_plot(ax1,Z_train,y_values,title)
        
            title="BBB Binary-classification Black Box"
            y_values=self.__dict_latent_spaces[comb]["binary_bbox"].predict(sub_ij_training_dataset.iloc[:,:-1])
            y_values=self.__dict_latent_spaces[comb]["label_encoder"].inverse_transform(y_values)
            
            ax2=self.__prepare_ls_class_scatter_plot(ax2,Z_train,y_values,title,comb)

            fig.suptitle(f"Plot of latent space of combination {comb}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            plt.show()
                
            return
            
        if (class_name!=None and class_name in self.__all_classes):
                
            for latent_space_comb in self.get_ls_of_class(class_name):

                # Create two subplots, side by side (1 row, 2 columns)
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
                
                sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb[0]]
                _, Z_train = self.__dict_latent_spaces[latent_space_comb[0]]["latent"].transform(sub_ij_training_dataset.iloc[:,:-1].values)

                title="MBB Multi-classification Black Box"
                y_values=self.__mbbox_model.predict(sub_ij_training_dataset.iloc[:,:-1])
                ax1=self.__prepare_ls_class_scatter_plot(ax1,Z_train,y_values,title)

                title="BBB Binary-classification Black Box"
                y_values=self.__dict_latent_spaces[latent_space_comb[0]]["binary_bbox"].predict(sub_ij_training_dataset.iloc[:,:-1])
                y_values=self.__dict_latent_spaces[latent_space_comb[0]]["label_encoder"].inverse_transform(y_values)
                ax2=self.__prepare_ls_class_scatter_plot(ax2,Z_train,y_values,title,latent_space_comb[0])
                
                fig.suptitle(f"Plot of latent space of combination {latent_space_comb[0]}")
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                plt.show()

    def plot_ls_training_losses(self,comb=None,class_name=None):

        """
            Function callable from outside to plot the curves of training and validation losses during generation of latent spaces.
    
            Args:
                comb: the specific combination involved in the plot
                class_name: the specific class involved in the combinatio to be plotted
        """
        
        if (comb==None and class_name==None):
            for latent_space_comb in self.__dict_latent_spaces:
                title = f"Plot training loss of latent space of combination {latent_space_comb}"
                sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb]
                self.__plot_ls_losses(self.__dict_latent_spaces[latent_space_comb]["losses"],title)
                
        if (comb!=None and comb in self.__class_combinations):
            sub_ij_training_dataset=self.__dict_train_datasets[comb]
            title=f"Plot training loss of latent space of combination {comb}"
            self.__plot_ls_losses(self.__dict_latent_spaces[comb]["losses"],title)
            return
                
        if (class_name!=None and class_name in self.__all_classes):
            for latent_space_comb in self.get_ls_of_class(class_name):
                title=f"Plot training loss of latent space of combination {latent_space_comb[0]}"
                sub_ij_training_dataset=self.__dict_train_datasets[latent_space_comb[0]]
                self.__plot_ls_losses(latent_space_comb[1]["losses"],title)

    
    def execute_testing_on_data(self,test_name,test_instances,change_f,max_f,filtering_mode="acceptable",debug=False):
        """
            This function call iteratively the methods to generate the counterfactuals and save all the results on external files
            specified by input parameters, in order to make a post-hoc analyis of the goodness of used methods.
            
            Args:
                test_name: the name to be used as suffix in the generation of the files
                test_instances: the list of instances involved during the test of the methods
                change_f: the list of feature that can be modifiable during the procedure of generation
                max_f: it is the maximum number of attribute that can be changed togheter from the change_f list
                filtering_mode: (optionally) it indicate the modality of possible filtering of counterfactuals 'acceptable','specific',default 'acceptable'
                debug: boolean value to print on console debug informations
            
        """
        # Create the directory tests if it doesn't exist
        dir_name = "tests"
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            print(f"Directory '{dir_name}' created.")

        #Create FILE1 new source of test_set with instance id
        ts=test_instances.copy()
        ts=ts.iloc[:,:-1]
        ts.insert(0, 'ID_TEST', range(1, len(ts) + 1))
        ts.to_csv("./"+dir_name+"/testset_"+test_name+".csv",index=False)

        #Create FILE2 intermediary informations about test_instance and its counterfactuals
        index_position=0
        df = pd.DataFrame(columns=['ID_TEST','orig_test_class','mbbox_test_class_pred','counter_class_desired',
                                   'ID_COUNTER','bbbox_counter_class_pred','mbbox_counter_class_pred','mmbox_counter_class_prob','counter_distance'])

        #Create FILE3 new counterfactuals_set with id: THIRD FILE
        id_counterfactual=1
        df_counterfactuals = pd.DataFrame()

        #SECOND COLUMN -- info related to the original class
        y_labels=test_instances.iloc[:,-1]

        #THIRD COLUMN -- info related to the prediction of mbbox of original instance
        y_mbbox_predictions=self.__mbbox_model.predict(test_instances.iloc[:,:-1])
        
       
        for index in tqdm(range(0, ts.shape[0]), desc="Processing test instances"):
            
            #index used to get the exact test_instance
            test_instance=ts.iloc[index:index+1,1:]
            
            test_instance_id=ts["ID_TEST"].iloc[index]

            ris=self.generate_counterfactuals(test_instance,change_f,max_f,counter_classes=-1,filtering_mode=filtering_mode,debug=debug)

            for label in ris.keys():

                #FOURTH COLUMN: label used to get the desired class of the counterfactual
                counter_class_desired=label
                
                if (ris[label]["all_counterfactuals"] is not None and not ris[label]["all_counterfactuals"].empty):
                    
                    gen_counterfactuals=ris[label]["all_counterfactuals"].copy()
                    
                    #EIGHTH COLUMN: probability prediction of multi-class classifier
                    probab=self.__mbbox_model.predict_proba(gen_counterfactuals)
                    max_prob=np.max(probab, axis=1)
                    #NINTH COLUMN: distance between test sample and generated counterfactuals
                    distances=self.cdist_sample_distance(test_instance,gen_counterfactuals)
    
                    #FIVETH COLUMN: id of generated counterfactual
                    gen_counterfactuals.insert(0,'ID_COUNTER',range(id_counterfactual,id_counterfactual+gen_counterfactuals.shape[0]))
                    
                    df_counterfactuals=pd.concat([df_counterfactuals,gen_counterfactuals])
    
                    for index_counterfactual in range(gen_counterfactuals.shape[0]):
    
                        #index_counterfactual used to get the exact counterfactual among the generated countefactuals
                        
                        #SIXTH COLUMN: counterfactual's class prediction of binary classifier 
                        counter_bbbox_pred=ris[label]["bbbox_predictions"][index_counterfactual]
                        #SEVENTH COLUMN: counterfactual's class prediction of multiclass classifier 
                        counter_mbbox_pred=ris[label]["mbbox_predictions"][index_counterfactual]
    
                        ## TO DEBUG
                        #print(f"ID_TEST: {test_instance_id}")
                        #print(f"orig_class_test: {y_labels[index]}")
                        #print(f"mbbox_class_pred_test: {y_mbbox_predictions[index]}")
                        #print(f"counter_class_desired: {label}")
                        #print(f"ID_COUNTER: {gen_counterfactuals['ID'][index_counterfactual]}")
                        #print(f"bbbox_class_counter_pred: {counter_bbbox_pred}")
                        #print(f"mbbox_class_counter_pred: {counter_mbbox_pred}")
                        #print(f"mmbox_counter_class_prob: {float(max_prob[index_counterfactual]):.4f}")
                        #print(f"counter_distance: {float(distances[0][index_counterfactual]):.4f}")
                        
                        df.loc[index_position]={"ID_TEST":test_instance_id,
                                                "orig_test_class":y_labels.iloc[index],
                                                "mbbox_test_class_pred":y_mbbox_predictions[index],
                                                "counter_class_desired": label,
                                                "ID_COUNTER":gen_counterfactuals["ID_COUNTER"][index_counterfactual],
                                                "bbbox_counter_class_pred":counter_bbbox_pred,
                                                "mbbox_counter_class_pred":counter_mbbox_pred,
                                                "mmbox_counter_class_prob": float(f"{max_prob[index_counterfactual]:.4f}"),
                                                "counter_distance":float(f"{distances[0][index_counterfactual]:.4f}")
                                               }
                        index_position=index_position+1
                    id_counterfactual=id_counterfactual+gen_counterfactuals.shape[0]
                else:
                    df.loc[index_position]={"ID_TEST":test_instance_id,
                                                "orig_test_class":y_labels.iloc[index],
                                                "mbbox_test_class_pred":y_mbbox_predictions[index],
                                                "counter_class_desired": label,
                                                "ID_COUNTER":None,
                                                "bbbox_counter_class_pred":None,
                                                "mbbox_counter_class_pred":None,
                                                "mmbox_counter_class_prob": None,
                                                "counter_distance":None
                                           }
                    index_position=index_position+1
        
        df_counterfactuals.to_csv("./"+dir_name+"/counterfactualset_"+test_name+".csv",index=False)
        df.to_csv("./"+dir_name+"/results_"+test_name+".csv",index=False)     
        
#pydoc.writedoc("MultiClass_CP_ILS")   
ens_latent_spaces=MultiClass_CP_ILS(None,None,None,None,"class",mode="1v1")