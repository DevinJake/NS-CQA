# NS-CQA: for JWS 2020 submission.
Our paper is published in JWS 2020 [1], which is 'Less is more: Data-efficient complex question answering over knowledge bases'.
We aim to solve the CQA task [2], targeting at answering factual questions through complex inferring over a realistic-sized KG of millions of entities.  

We could learn the details of the CQA dataset [here](https://amritasaha1812.github.io/CSQA/download_CQA/).  

All the materials required for running the KG sever, training the model, and testing in this task could be downloaded from the [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing).  

We should follow the folder structure in the data link, and place the files in the corresponding location under the `data` folder.
  
Following this README, we will instruct how to use the relevant data from the data link. 

---

The questions in the CQA could be categorized into seven groups.  
The typical examples of these seven question types are displayed in the following table.  
  
   Question Type   |  Question                              |  KB artifacts          |  Action Sequence   |        Answer
   -------- | --------  | -------- | --------| -------- |
   Simple | Which administrative territory is Danilo Ribeiro an inhabitant of? | E1: Danilo Ribeiro <br> R1: country of citizenship <br> T1: administrative territory | Select(E1, R1, T1) | Brazil 
   Logical | Which administrative territories are twin towns of London but not Bern? | E1: London <br> E2: Bern <br> R1: twinned adm. body <br> T1: administrative territory | Select(E1, R1, T1) <br> Diff(E2, R1, T1) | Sylhet, Tokyo, Podgorica, <br> Phnom Penh, Delhi, <br> Los Angeles, Sofia, New Delhi, ...
   Quantitative | Which sports teams have min number of stadia or architectural structures as their home venue? | R1: home venue <br> T1: sports team <br> T2: stadium <br> T3: architectural structure | SelectAll(T1, R1, T2) <br> SelectAll(T1, R1, T3) <br> ArgMin() | Detroit Tigers, Drbak-Frogn IL, <br> Club Sport Emelec, Chunichi Dragons, ...
   Comparative | Which buildings are a part of lesser number of architectural structures and universities than Midtown Tower? | E1: Midtown Tower <br> R1: part of <br> T1: building <br> T2: architectural structure <br> T3: university | SelectAll(T1, R1, T2) <br> SelectAll(T1, R1, T3) <br> LessThan(E1) | Amsterdam Centraal, Hospital de Sant Pau, <br> Budapest Western Railway Terminal, <br> El Castillo, ...
   Verification | Is Alda Pereira-Lemaitre a citizen of France and Emmelsbull-Horsbull? | E1: Alda Pereira-Lemaitre <br> E2: France <br> E3: Emmelsbull-Horsbull <br> R1: country of citizenship <br> T1: administrative territory | Select(E1, R1, T1) <br> Bool(E2) <br> Bool(E3) | YES and NO respectively
   Quantitative Count | How many assemblies or courts have control over the jurisdiction of Free Hanseatic City of Bremen? | E1: Bremen <br> R1: applies to jurisdiction <br> T1: deliberative assembly <br> T2: court | Select(E1, R1, T1) <br> Union(E1, R1, T2) <br> Count() | 2
   Comparative Count | How many art genres express more number of humen or concepts than floral painting? | E1: floral painting <br> R1: depicts <br> T1: art genre <br> T2: human <br> t3: concept | SelectAll(T1, R1, T2) <br> SelectAll(T1, R1, T3) <br> GreaterThan(E1) <br> Count() | 8

---
We first clone the project:
```
git clone https://github.com/DevinJake/NS-CQA.git
``` 
, and we could download a project `NS-CQA`.

---
## CQA dataset
Now we will talk about how to training and testing our proposed model on CQA dataset.  
### 1. Experiment environment.
 (1). Python = 3.6.4  
 (2). PyTorch = 1.1.0  
 (3). TensorFlow = 1.14.0  
 (4). tensorboardX = 2.1  
 (5). ptan = 0.4  
 (6). flask = 1.1.2  
 (7). requests = 2.24.0  
  
### 2. Accessing knowledge graph.
 (1). Assign the IP address and the port number for the KG server.

 Manually assign the IP address and the port number in the file of the project `NS-CQA/BFS/server.py`.
   
 Insert the host address and the post number for your server in the following line of the code:  
 ```
 app.run(host='**.***.**.**', port=####, use_debugger=True)
 ```

 Manually assign the IP address and the port number in the file of the project `NS-CQA/S2SRL/SymbolicExecutor/symbolics.py`.  
 Insert the host address and the post number for your server in the following ***three*** lines in the `symbolics.py`: 
 ```
 content_json = requests.post("http://**.***.**.**:####/post", json=json_pack).json()
 ```
  
 (2). Run the KG server.  
 Download the bfs data `bfs_data.zip` from the provided [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing).   
 We need to uncompress the file `bfs_data.zip` and copy the three pkl files into the project folder `NS-CQA/data/bfs_data`.   
 Run the project file `NS-CQA/BFS/server.py` to activate the KG server for retrieval: 
 ```
 python server.py
 ```
  
### 3. Training the neural generator.
 (1). Load the pre-trained models.
 By using a breadth-first-search (BFS) algorithm, we generated pseudo-gold action sequences for a tiny subset of questions and pre-trained the model by Teacher Forcing with the help of these pairs of questions and action sequences.
 Therefore, we will further train the neural generator by using Reinforcement learning.   
 
 We could download the pre-trained RL model `pre_bleu_0.956_43.zip`, uncompress it, and place it in the project folder `NS-CQA/data/saves/pretrained`.  
 
 If you want to pre-train the models by yourself, you could:  
 1. Download the 'data\auto_QA_data\mask_even_1.0%\PT_train_INT.question' and 'data\auto_QA_data\mask_even_1.0%\PT_train_INT.action' from the [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing). Put them under the folder 'mask_even_1.0%'. The files are the pseudo-gold annotations that were formed by using the BFS algorithm.
2. Run python file 'S2SRL\train_crossent.py'.
3. Under the folder 'data\saves\pretrained' you could find the models, which are trained under the teacher-forcing paradigm. The models are named following the format 'pre_bleu_TestingScore_numberOfEpochs.dat'. Normally we chose the model with the highest bleu testing score as the pre-trained model for following RL training. Or you could choose whatever model you like. 
4. The performance of the chosen model might have a tiny difference between the uploaded pre-trained model 'pre_bleu_0.956_43.dat'. We recommend you to choose 'pre_bleu_0.956_43.dat' for re-implementation. 

 We also provided the code for the **BFS** algorithm.
 If you are interested, you could:
 1. Download the 'data\official_downloaded_data\10k\train_10k.zip' from the [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing). Uncompress the zip file and put them under the folder 'train_10k'. The files are the provided 10K samples of the CQA dataset.
 
 2. Under the folder `NS-CQA/S2SRL/SymbolicExecutor`, we run the python file to search for the pseudo-gold annotations for the `simple` type questions by using the BFS algorithm: 
 ```
 python auto_symbolic_simple.py
 ```
 The pseudo-gold annotations would be stored in the file `NS-CQA/data/annotation_logs/jws_simple_auto.log`. 
 Similarly, we could get `jws_logical_auto.log` and `jws_count_auto.log` by running `auto_symbolic_logical.py` and `auto_symbolic_count.py`.
 We provide the BFS codes for these three types of the questions for demonstration.  
 
 (2). Train the neural generator.  
 In the project folder `NS-CQA/S2SRL`, we run the python file to train the MAML model: 
 ```
 python train_scst_cher.py
 ```
 The trained neural generator and the corresponding action memory would be stored in the folder `NS-CQA/data/saves/rl_cher`.   
 
### 4. Testing.
  (1). Load the trained model.  
  The trained models will be stored in the folder `NS-CQA/data/saves/rl_cher`.  
  We have saved a trained CQA model `epoch_022_0.793_0.730.zip` in this folder, which could leads to the SOTA result.  
  We could download such model from the [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing), uncompress it, and place it under the corresponding project folder.  
  When testing the model, we could choose a best model from all the models that we have trained, or simply use our trained model `epoch_022_0.793_0.730.dat`.  
  
  (2). Load the testing dataset.  
  We also have processed the testing dataset `SAMPLE_FINAL_INT_test.question` (which is 1/20 of the full testing dataset) and `FINAL_INT_test.question` (which is the full testing dataset), and saved them in the folder `NS-CQA/data/auto_QA_data/mask_test`.  
  We could download the files from the [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing) and put them under the folder `NS-CQA/data/auto_QA_data/mask_test` in the project.  
  
  (3). Test.  
  In the project file `NS-CQA/S2SRL/data_test.py`, we could change the parameters to meet our requirement.  
  In the command line: 
  ```
  sys.argv = ['data_test.py', '-m=epoch_022_0.793_0.730.dat', '--cuda', '-p=final_int',
                '--n=rl_cher', '--att=0', '--lstm=1',
                '--int', '-w2v=300', '--beam_search']
  ```
  , we could change the following settings.
    
  If we want to use the entire testing dataset to get the QA result, we should set `-p=final_int`.  
  Otherwise, we could set `-p=sample_final_int` to test on the subset of the testing dataset to get an approximation testing result with less time.  
  Based on our observation, the testing results of the entire testing dataset are always better than those of the subset.  
  
  If we want to use the models stored in the named folder `NS-CQA/data/saves/rl_cher`, we set `--n=rl_cher`.  
  If we want to use our saved CQA model `net_***.dat` in the named folder to test the questions, we set `-m=net_***.dat`.  
  
  After setting, we run the file `NS-CQA/S2SRL/data_test.py` to generate the action sequence for each testing question:
  ```
  python data_test.py
  ```
  We could find the generated action sequences in the folder where the model is in (for instance `NS-CQA/data/saves/rl_cher`), which is stored in the file `final_int_predict.actions` or `sample_final_int_predict.actions`.   
  
  (4). Calculate the result.   
  Firstly, we should download the files `CSQA_ANNOTATIONS_test_INT.json` from the [data link](https://drive.google.com/drive/folders/11HM--PcWxGicHnwMRTmgZ3GWCdugxiNC?usp=sharing) and put it into the folder `NS-CQA/data/auto_QA_data/` of the project, which is used to record the ground-truth answers of each question.  
   
  After generating the actions, we could use them to compute the QA result.  
  For example, we use the saved models to predict actions for the testing questions, and therefore generate a file `NS-CQA/data/saves/rl_cher/final_int_predict.actions` to record the generated actions for all the testing questions.  
    
  Then in the file `NS-CQA/S2SRL/SymbolicExecutor/calculate_sample_test_dataset.py`, we set the parameters as follows.  
  This line of code
  ```
calculate_RL_or_DL_result('rl_cher', withint=True)
  ```
will finally call the function `transMask2Action()` to compute the accuracy of the actions stored in the file `NS-CQA/data/saves/rl_cher/final_int_predict.actions`.  
  
  We could change the path of the generated file in the following line in the function `transMask2Action()` if we want:
  ```
  with open(json_path, 'r') as load_f, \
            open("../../data/saves/rl_cher/final_int_predict.actions", 'r') as predict_actions, \
            open(question_path, 'r') as RL_test:
  ```
  
  Then we run the file `NS-CQA/S2SRL/SymbolicExecutor/calculate_sample_test_dataset.py` to get the final result:
  ```
  python calculate_sample_test_dataset.py
  ```
  
  The result will be stored in the file `NS-CQA/data/auto_QA_data/test_result/rl_cher.txt`.  
  
  
## WebQSP dataset
Now we will talk about how to training and testing our proposed model on WebQSP(WebQuestions Semantic Parsest) dataset.  
### 1. Experiment environment.
 (1). Python = 3.6.4  
 (2). PyTorch = 1.1.0  
 (3). TensorFlow = 1.14.0  
 (4). tensorboardX = 2.1  
 (5). ptan = 0.4  
 (6). flask = 1.1.2  
 (7). requests = 2.24.0  
  
### 2. Accessing knowledge graph.
 (1). Assign the IP address and the port number for the KG server.
 Manually assign the IP address and the port number in the file of the project NS-CQA/webqspUtil/server.py
 Insert the host address and the post number for your server in the following line of the code:
 
 ```
 app.run(host='**.***.**.**', port=####, use_debugger=True)
 ```

 Manually assign the IP address and the port number in the file of the project `NS-CQA/S2SRL/SymbolicExecutor/symbolics_webqsp_novar.py`.  
 Insert the host address and the post number for your server in the following ***three*** lines in the `symbolics.py`: 
 ```
 content_json = requests.post("http://**.***.**.**:####/post", json=json_pack).json()
 ```
  
 (2). Run the KG server.  
 The kg we used is the subgraph about webQSP of freebase, you can download the file `webQSP_freebase_subgraph.zip` 
 form the provided [data link](https://drive.google.com/file/d/1ZCkE51pG70X0Uwlr-HcI16X5pB7pFj-h/view?usp=sharing),
 the final path is : `data/webqsp_data/webQSP_freebase_subgraph.json`.
 Run the project file `NS-CQA/webqspUtil/server.py` to activate the KG server:

 ```
 python server.py
 ```
  
### 3. Training the neural generator.
 (1). Prepare the data.  
 The WebQSP dataset can be download from the [data link](https://drive.google.com/file/d/1ZCkE51pG70X0Uwlr-HcI16X5pB7pFj-h/view?usp=sharing). We process the original dataset to
 train and test in our experiment. 
 The processed data we use can be download from the [data link](https://drive.google.com/file/d/18ykhC4x8P_1AUnQBzeGHxIi6F0jbDPwI/view?usp=sharing). WE could place the processed
 pre-train data `PT_train.question`, `PT_train.action`, `PT_test.question`, `PT_test.action` in the folder `NS-CQA/data/webqsp_data/mask`,
 place vocab file `share.webqsp.question` and final data `final_webqsp_train_RL.json`, `final_webqsp_test_RL.json` in `NS-CQA/data/webqsp_data`,
 
 (2). Load the pre-trained models.  
 You can run the project file `NS-CQA/S2SRL/train_crossent_webqsp.py` to train your own pre-train model:
  ```
 python train_crossent_webqsp.py
 ```
 or download the pre-trained RL model [epoch_030_0.995_0.966.dat](https://drive.google.com/file/d/1Jh68QFELDsp5B9QP0EiZHMLkTcMYk6_B/view?usp=sharing), uncompress it, 
 and place it in the project folder `NS-CQA/data/webqsp_saves/pretrained`.  
 
 
 (2). Train the neural generator.  
 In the project folder `NS-CQA/S2SRL`, we run the python file to train the model: 
 ```
 python train_scst_cher_webqsp.py
 ```
 The trained neural generator and the corresponding action memory would be stored in the folder `NS-CQA/data/webqsp_saves/rl_cher`.   
 
### 4. Testing.
  (1). Load the trained model.  
  The trained models will be stored in the folder `NS-CQA/data/webqsp_saves/rl_cher`.  
  We have saved a trained CQA model `epoch_000_0.838_0.000.dat` in this folder, which could leads to the SOTA result.  
  We could download such model from the [data link](https://drive.google.com/file/d/1Jh68QFELDsp5B9QP0EiZHMLkTcMYk6_B/view?usp=sharing), uncompress it,
  and place it under the corresponding project folder.  
  When testing the model, we could choose a best model from all the models that we have trained, or simply use our trained model `epoch_000_0.838_0.000.dat`.  
  
  (2). Load the testing dataset.  
  You can find `final_webqsp_test_RL.json` in `NS-CQA/data/webqsp_data` as our test data.
  
  (3). Test.  
  In the project file `NS-CQA/S2SRL/data_test_RL_webqsp.py`, we could change the parameters to meet our requirement.  
  In the command line: 
  ```
          sys.argv = ['data_test_RL_webqsp.py', '-m=epoch_000_0.838_0.000.dat', '-p=rl', '--n=rl_cher', '--att=0', '-w2v=300', '--lstm=1']
  ```
  , we could change the following settings.
  
  If we want to use the models stored in the named folder `NS-CQA/data/saves/rl_cher`, we set `--n=rl_cher`.  
  If we want to use our saved WebQSP model `net_***.dat` in the named folder to test the questions, we set `-m=net_***.dat`.  
  
  After setting, we run the file `NS-CQA/S2SRL/data_test_RL_webqsp.py` to generate the action sequence for each testing question:
  ```
  python data_test_RL_webqsp.py
  ```
  We could find the generated action sequences in the folder where the model is in (for instance `NS-CQA/data/webqsp_saves/rl_cher`),
  which is stored in the file `rl_predict.actions`.   
  

 #### References:  
 [1]. Yuncheng Hua, Yuan-Fang Li, Guilin Qi, Wei Wu, Jingyao Zhang, and Daiqing Qi. 2020. Less is more: Data-efficient complex question answering over knowledge bases. In Journal of Web Semantics 65 (2020): 100612.
 
 [2]. Amrita Saha, Vardaan Pahuja, Mitesh M Khapra, Karthik Sankaranarayanan, and Sarath Chandar. 2018. Complex sequential question answering: Towards learning to converse over linked question answer pairs with a knowledge graph. In ThirtySecond AAAI Conference on Artificial Intelligence.
 
 #### Cite as:
 > Hua, Y., Li, Y. F., Qi, G., Wu, W., Zhang, J., & Qi, D. (2020). Less is more: Data-efficient complex question answering over knowledge bases. Journal of Web Semantics, 65, 100612.
 