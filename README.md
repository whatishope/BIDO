# BIDO
This code belongs to "BIDO: A Unified Approach to Address Obfuscation and Concept Drift Challenges in Image-based Malware Detection".
# Author
Author:  Junhui Li1, Chengbin Feng2, Zhiwei Yang1, Qi Mo1 and Wei Wang1.
# Institution
Institution1：Software School, Yunnan University, Kunming, Yunnan 650091 China.
Institution2：School of Information Systems, University of New South Wales, Sydney, NSW 2052 Australia.
# Dataset Availability
Due to the large size of the datasets, we provide public access via cloud storage and detailed usage instructions in this repository.
## Cloud Storage
The complete Data-Ideal and Data-Obfu datasets are hosted on OneDrive:https://1drv.ms/f/c/90ae1e5185e3d1c1/IgDhovkur4wcSbDRiadcg66nAVDRM9BNRXRN5ftU2kKnIcY?e=skQjJD
## Usage Instructions
This repository provides detailed instructions on dataset structure, preprocessing, and usage in the sections below.
# Dataset Description
This project uses two Android malware datasets designed to evaluate both ideal learning conditions and robustness under obfuscation.
## Data-Ideal Dataset
### Composition
The dataset contains a total of 24,830 Android applications, including 12,375 malicious samples and 12,455 benign samples.
### Labeling Strategy
Labels are assigned based on VirusTotal scan results. An application is labeled as benign if it receives zero detections, and labeled as malicious if it is flagged by more than four antivirus engines.
### Data Sources
APK files are collected from multiple widely used and authoritative sources, including Google Play, AndroZoo, and the CICMalDroid2020 benchmark dataset.
## Data-Obfu Dataset
### Generation Process
This dataset is derived from the Data-Ideal dataset by applying six obfuscation techniques sequentially to each application using the Obfuscapk tool.
### Obfuscation Techniques
The applied techniques include Class and Method Renaming, Resource String Encryption, Control Flow Obfuscation, New Alignment, New Signature, and Junk Code Insertion.
### Final Dataset Size
After removing applications that encountered errors during the obfuscation process, the final dataset consists of 12,088 malicious samples and 11,044 benign samples.




