# Analiza e datasetit Data Science Job Postings & Skills (2024)
Repo per analizen e datasetit me job postings dhe aftesite e kerkuara

Fazat e perfshira gjate analizes: 
1. Pergaditja e modelit the paraprocesimi i te dhenave
2. Trajnimi i modelit
3. Analizimi dhe reevaluimi duke perdorur teknikat e Machine Learning


Ketu do dokumentoj te gjithe hapat dhe fazat e perdorimit te teknikave te Machine Learning per te analizuar dhe nxjerr njohuri nga dataseti ne fjale. Do te analizojme dhe pershkoj te gjitha hapat e nevojshem per te arritur analize adekuate me rezultate me te sakta dhe kualitative.

Gjate analizes sone do te mundohemi te nxjerrim informata per 
1. lidhshmerine mes aftesive dhe vendeve te punes apo niveleve te punes
2. te parashikojme levelin e punes ne baze te aftesive (job_skills) dhe pershkrimit (job_summary)
3. me ane te NLP te nxjerrim info per pagen e pozitave te punes dhe te arrijme te parashikojme pagen per nje vend te ri te punes
4. te kategorizojme vendet e punes me ane te clustering ne teknike dhe jo-teknike

Te gjitha fazat qe do tí implementojme dhe konkluzionet e nxjerra bazohen ne qualitetin e datasetit dhe veprimet qe kane kuptim per kete dataset.

# FAZA 1: Preparing the model

**1_getdataset.py**
Le te fillojme me shkarkimin e datasetit nga kaggle, e kemi realizuar kete duke i shkarku files direkt nga kaggle API, per ta bere kete na duhet te gjenerojme nje token nga kaggle->settings nga ku do te ju shkarkohet nje kaggle.json file qe permban tokenin, lokacioni baze ku duhet vendosur eshte ne C:/Users/[useri_juaj]/.kaggle
nese e beni run 1_getdataset.py atehere do te shkarkohen datasetet ne direktorin data ne kuarder te prjektit. 

**2_preprocessing.py**
Bazuar ne analizen fillestare kemi verejtur se duhet te bejme paraprocesim te te dhenave qe fillimisht te kombinojme datasetet e ndara ne baze te vetise se perbashket qe kane (job_link) dhe me pas ti pastrojme te dhenat.
Si fillim duhet te fillojme me fazen e pare: njohjen e datasetit dhe pergaditjen e modelit 

Dataseti permban 3 fajll te ndryshem nga te cilet verejme edhe 3 objekte te dhenash me attributet e tyre
Job Posting
Job Skill
Job Summary
Atributet e nje Job Posting objekt jane: [ 'job_link', 'last_processed_time', 'last_status', 'got_summary', 'got_ner', 'is_being_worked', 'job_title', 'company', 'job_location', 'first_seen', 'search_city', 'search_country', 'search_position', 'job_level', 'job_type' ]

Atributet e Job Summary jane: [ 'job_link', 'job_summary' ]

Attributet of Job Skills jane: [ 'job_link', 'job_skills' ]

Disa teknika te paraprocesimit qe i kemi perdorur jane largimi i rekordeve te zbrazta, eliminimi i rekordeve te dyfishta, krijimi i vetive te reja duke u bazuar ne ato ekzistuese, normalizimi i fushave tekstuale siq eshte job_summary duket i kthyer te gjitha fjalet ne lowercas, standardizimi i job_location ne qytet/shtet. Shumica e ketyre veprimeve jane kryer ne preprocessing.py

**salary-analysis/nlp_salary_extract**
ne kete skript kemi realizuar nxjerrjen e te dhenave per pagen nga pershkrimi i vendit te punes per ato rekorde te cilat e kane pas te cekur pagen, kjo eshte arritur duke perdorur NLP, per shkak te madhesise se datasetit merr kohe generimi i fajlit te ri i cili permban edhe attributin e pages, megjithate per lehtesim e kemi attach edhe fajllin e generuar ne /data/classified_job_postings_with_salary.csv 
Funksionet ndihmese per nxjerrjen e pages dhe pastaj paraprocesimin se a eshte kjo page vjetore, mujore apo me ore dhe llogaritjen e tyre jane bere permes funksioneve ne helper.py

# FAZA 2: Trajnimi i modelit

Ne kete faze kemi filluar te eksplorojme me teknikat e ndryshme te machine learning ne menyre qe te arrijme te gjeme pergjigje per detyrat qe i kemi parashtru me larte.

## Lidhshmerine mes aftesive dhe vendeve te punes apo niveleve te punes e kemi gjet duke perdor LogisticRegression dhe TF-IDF per vektorizimin e aftesive. [execute 3_regression.py]

Evaluimin e modelit e kemi bere permese Precision score dhe confusion matrix 

Precision Score: 0.903411946219969
Confusion Matrix:
[[  21  217]
 [   4 2201]]

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/123add1f-08b2-4f31-ab47-e975a9345d8e)

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/cabe0ddc-70c8-446d-9d65-e04fd994ccb6)

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/87878446-2a83-42f3-9697-501c4d5c5ae8)

## te parashikojme levelin e punes ne baze te aftesive (job_skills) dhe pershkrimit (job_summary) [execute 4_regression.py]
ketu kemi mar si target job_level dhe me pas me feature engineering kemi nxjerr info mbi skills the job summary dhe i kemi kombinu me pas kemi perdore regresionin per me arrite me parashiku job_level

Si mates te suksesit kemi perdore Mean Squared Error

Mean Squared Error: 0.09125216223132329
MSE scores for the 5 folds: [0.08241941 0.12168425 0.12421318 0.0625263  0.04852547]
Average MSE: 0.08787372524941817

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/5228c1a4-dbaa-44d1-9c4a-ffc512a09a3b)

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/1e7d2ece-8af6-4fc8-b2d1-99d6b8d2c526)

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/3bdeb01c-c896-4e95-9979-c5ae65480de9)

## te kategorizojme vendet e punes me ane te clustering ne teknike dhe jo-teknike

dataseti jon nuk permban ndonje informat lidhur me natyren e ofertave te punes dhe nuk i klasifikon ato sipas natyres se detyrave qe duhet kryer, andaj ne kemi vendosur te perdorim clustering algoritem qe te arrijme te nje klasifikim/grupim i job posts ne teknike dhe jo teknike.

Per clstering kemi perdor K-Means

cluster
0    11688
1      529

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/499aee67-3915-4274-8437-d237e7426486)

FAZA 3

Per fazen tre mendoj te analizoj me thelle algoritmet dhe paraprocesimin qe duhet te beje qe te arrij parashikimin e pages ne menyre me te sakte.
Kam perdorur neural network analysis ne menyre qe te arrij rezultate me te mira siq shihen me poshte. 

kemi vendosur nje threshold te caktuar dhe nese performance e modelit bie atehere vazhdojme em ritrajnimin e modelit 

sa i perket ML tools kemi perdorur TensorFlow nga Google Brains dhe modelin neural networks 

![image](https://github.com/krenareshalarrmoku/machinelearning/assets/165852868/3c2bf0b1-dae3-46a6-ab95-bb277ff41f2f)
 





