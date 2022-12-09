# LTE 
The code for ICDE 2023 paper: *Learn to Explore: on Bootstrapping Interactive Data Exploration with Meta-learning*
    
    
## Main Package
* pytorch
* sklearn
* numpy
* pandas

## How to use?
1.	All runtime parameters are set in **GlobalConfigs.py**

 * ***dataspace_configs*** is used to construct Dataspace models, where subspaces are predefined on ***dataspace_configs['online_split_list']/['offline_split_list']*** .
 
 * ***taskGenerate_configs***  is used to generate meta-tasks for meta-training.

 * ***mamexplore_configs*** is used to train meta-learners.

 * ***offlineTaskGenerate_configs*** is used to generate te st tasks.
 
2.	Training

 ```bash
cd Train_code/
python3 Main.py
 ```
 *	The meta-tasks will be stored in ***train_task_root_x_x_x_x/***
	
3.	Testing
 ```bash
 cd Test_code/
 python3 Main_test.py #generate test tasks
 python3 Test.py #testing
 python Result.py #show result
  ```
  * The modes of the test tasks can be freely configured in ***Main_test.py*** (*dict format*), for example:
  ```json
  mode_list={
  '1':{'attr_list':[['price', 'powerPS']],  'u_flag':None}} #2D
  mode_list = {
  '1':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g'],['ra', 'dec']], 'u_flag': None, 'dim':   [1,1,1], 'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]},
  '2':{'attr_list': [['rowc', 'colc'],['sky_u','sky_g']],'u_flag': None,'dim': [2,1], 'reg': [[1, 15], [1, 20], [1, 25], [1, 30]]}} #4D-8D
  ```
  
  * The test task will be stored in ***test_offline_task_x_x_x_modex/***
  * For the high-dimensional case, you can use ***Main_test_Random.py***, which can generate test tasks in large batches, randomised.
 * The test results are saved in the file: ***solvetest_offline_task_x_x_x_modex_x_train_task_root_x_x_x_x_nomal***
  

Thanks to the project: 
> @MAMO: https://github.com/dongmanqing/Code-for-MAMO
> @CTGAN: https://github.com/sdv-dev/CTGAN

