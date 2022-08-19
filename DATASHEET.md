# DIVOTrack: A Cross-View Dataset for Multi-Human Tracking in DIVerse Open Scenes

We present **DIVOTrack**: a new cross-view multi-human tracking dataset for **DIV**erse **O**pen scenes with dense tracking pedestrians in realistic and non-experimental environments If you use this dataset, please acknowledge it by citing the original paper:

```
@article{wangdivotrack,
  title={DIVOTrack: A Cross-View Dataset for Multi-Human Tracking in DIVerse Open Scenes},
  author={Wang, Gaoang and Hao, Shengyu and Zhan, Yibing and Liu, Peiyuan and Liu, Zuozhu and Song, Mingli and Hwang, Jenq-Neng},
  year={2022}
}
```

## Motivation


1. **For what purpose was the dataset created?** *(Was there a specific task in mind? Was there a specific gap that needed to be filled? Please provide a description.)*
    
    We propose a novel cross-view multi-human tracking dataset, namely *DIVOTrack*, which is more realistic and diverse, has more tracks, and incorporates moving cameras. Cross-view multi-human tracking tries to link human subjects between frames and camera views that contain substantial overlaps. Although cross-view multi-human tracking has received increased attention in recent years, existing datasets still have several issues, including 1) missing real-world scenarios, 2) lacking diverse scenes, 3) owning a limited number of tracks, 4) comprising only static cameras, and 5) lacking standard benchmarks, which hinders the exploration and comparison of cross-view tracking methods.

1. **Who created this dataset (e.g., which team, research group) and on behalf of which entity (e.g., company, institution, organization)?**
    
    This dataset was created by Gaoang Wang, Shengyu Hao, Yibing Zhan, Peiyuan Liu, Zuozhu Liu, Mingli Song and Hwang, Jenq-Neng. At the time of creation, Shengyu Hao and Peiyuan Liu were graduate students at Zhejiang University (ZJU), and Gaoang Wang, Zuozhu Liu and Mingli Song are Professors of Zhejiang University (ZJU), Yibing Zhan is from JD Explore Academy, Hwang Jenq-Neng is a professor of University of Washington (UW).


3. **Who funded the creation of the dataset?** *(If there is an associated grant, please provide the name of the grantor and the grant name and number.)*
    
    Funding for Gaoang Wang was provided by National Natural Science Foundation of China. 

4. **Any other comments?**
    
    None.





## Composition


1. **What do the instances that comprise the dataset represent (e.g., documents, photos, people, countries)?** *(Are there multiple types of instances (e.g., movies, users, and ratings; people and interactions between them; nodes and edges)? Please provide a description.)*
    
    Each instance is people, our DIVOTrack contains videos that are collected by two mobile cameras and one unmanned aerial vehicle.

2. **How many instances are there in total (of each type, if appropriate)?**
    Our DIVOTrack contains ten different types of scenarios and 550 cross-view tracks.
    We count the number of bounding boxes and tracks in the training and testing sets for each scene.
    The whole DIVOTrack dataset has 560K boxes, of which 270K boxes belong to the training set, while the rest belongs to the test. The different colors of each bar represent different views.
    We count the number of tracks from 60 videos. We can observe a large variant in different scenes on the number of tracks. (More details can see in paper ```Section 3.3```)

3. **Does the dataset contain all possible instances or is it a sample (not necessarily random) of instances from a larger set?** *(If the dataset is a sample, then what is the larger set? Is the sample representative of the larger set (e.g., geographic coverage)? If so, please describe how this representativeness was validated/verified. If it is not representative of the larger set, please describe why not (e.g., to cover a more diverse range of instances, because instances were withheld or unavailable).)*
    
    It is a sample of all possible samples. is more realistic and diverse, has more tracks, and incorporates moving cameras. Accordingly, a standardized benchmark is built for cross-view tracking, with clear split of training and testing set, publicly accessible detection, and standard cross-view tracking evaluation metrics. With the proposed dataset and benchmark, the cross-view tracking methods can be fairly compared in the future, which will improve the development of cross-view tracking techniques.


4. **What data does each instance consist of?** *(``Raw'' data (e.g., unprocessed text or images)or features? In either case, please provide a description.)*
    
    Each instance consists of videos contains frame ID, person ID, x_center, y_center, w, h. The format of the annotation is:

    ```
    Frame ID, person ID, x, y, w, h
    0 1 0.975781 0.333796 0.033854 0.163889
    0 3 0.628646 0.763889 0.054167 0.246296
    0 4 0.195833 0.716667 0.059375 0.233333
    ...
    ```
    Where x_center, y_center, w, h are the center coordinate, weight and height of the bounding box by normalization.

5. **Is there a label or target associated with each instance? If so, please provide a description.**
    
    We have ten scenes and each scene has three videos. The person in the same scene has the same ID. 


6. **Is any information missing from individual instances?** *(If so, please provide a description, explaining why this information is missing (e.g., because it was unavailable). This does not include intentionally removed information, but might include, e.g., redacted text.)*
    
    In some cases, we remove some person boxes which are very small and hard to detect.


7. **Are relationships between individual instances made explicit (e.g., users' movie ratings, social network links)?** *( If so, please describe how these relationships are made explicit.)*
    
    Scenes are largely unrelated.


8. **Are there recommended data splits (e.g., training, development/validation, testing)?** *(If so, please provide a description of these splits, explaining the rationale behind them.)*
    
    We provide a training/validation/testing split; we split training set and test set half-to-half for each scene. For example, the first half of the video is test set and rest is training set.

    **Warning:** If you *do* use any of the data for training/testing, either in a cross-validation setup or otherwise, you may wish to be careful to ensure that documents on very similar topics are always in the same split. For instance, there are two documents about Leslie Feinberg in the dataset; you should ensure that these are in the same split or evaluation scores are likely to be inflated.


9. **Are there any errors, sources of noise, or redundancies in the dataset?** *(If so, please provide a description.)*
    
    There are almost certainly some errors in annotation. We did our best to minimize these, but some certainly remain. There are a few error mosaic part on the ground and incorrect boxes, but the impact of these errors are limited.  

10. **Is the dataset self-contained, or does it link to or otherwise rely on external resources (e.g., websites, tweets, other datasets)?** *(If it links to or relies on external resources, a) are there guarantees that they will exist, and remain constant, over time; b) are there official archival versions of the complete dataset (i.e., including the external resources as they existed at the time the dataset was created); c) are there any restrictions (e.g., licenses, fees) associated with any of the external resources that might apply to a future user? Please provide descriptions of all external resources and any restrictions associated with them, as well as links or other access points, as appropriate.)*
    
    The dataset is self-contained.


11. **Does the dataset contain data that might be considered confidential (e.g., data that is protected by legal privilege or by doctor-patient confidentiality, data that includes the content of individuals' non-public communications)?** *(If so, please provide a description.)*
    
    No; all raw data in the dataset is from public scenes.


12. **Does the dataset contain data that, if viewed directly, might be offensive, insulting, threatening, or might otherwise cause anxiety?** *(If so, please describe why.)*
    
    Our dataset do not contain any information which may caused anxiety.


13. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
    Yes, all videos relate to real people (except the fan-fiction articles).


14. **Does the dataset identify any subpopulations (e.g., by age, gender)?** *(If so, please describe how these subpopulations are identified and provide a description of their respective distributions within the dataset.)*
    
    This is not explicitly identified, though all of the scenes contains people.


15. **Is it possible to identify individuals (i.e., one or more natural persons), either directly or indirectly (i.e., in combination with other data) from the dataset?** *(If so, please describe how.)*
    
    No, our dataset do not have personal information. Besides, we blur each face in videos.


16. **Does the dataset contain data that might be considered sensitive in any way (e.g., data that reveals racial or ethnic origins, sexual orientations, religious beliefs, political opinions or union memberships, or locations; financial or health data; biometric or genetic data; forms of government identification, such as social security numbers; criminal history)?** *(If so, please provide a description.)*
    
    Our dataset do not contain any sensitive data.


17. **Any other comments?**
    
    None.





## Collection Process


1. **How was the data associated with each instance acquired?** *(Was the data directly observable (e.g., raw text, movie ratings), reported by subjects (e.g., survey responses), or indirectly inferred/derived from other data (e.g., part-of-speech tags, model-based guesses for age or language)? If data was reported by subjects or indirectly inferred/derived from other data, was the data validated/verified? If so, please describe how.)*
    
    The data was collected in public areas and we annotated them by hand.

2. **What mechanisms or procedures were used to collect the data (e.g., hardware apparatus or sensor, manual human curation, software program, software API)?** *(How were these mechanisms or procedures validated?)*
    
    Two mobile phone and one UAV device.


3. **If the dataset is a sample from a larger set, what was the sampling strategy (e.g., deterministic, probabilistic with specific sampling probabilities)?**
    
    See answer to question #2 in [Composition](#composition).


4. **Who was involved in the data collection process (e.g., students, crowdworkers, contractors) and how were they compensated (e.g., how much were crowdworkers paid)?**
    
    All collection and annotation was done by the seven authors.


5. **Over what timeframe was the data collected?** *(Does this timeframe match the creation timeframe of the data associated with the instances (e.g., recent crawl of old news articles)?  If not, please describe the timeframe in which the data associated with the instances was created.)*
    
    The dataset was collected in the fall of 2020, which does not necessarily reflect the timeframe of the data collected.


6. **Does the dataset relate to people?** *(If not, you may skip the remaining questions in this section.)*
    
    Yes; all videos of each scene are related to people.


7. **Did you collect the data from the individuals in question directly, or obtain it via third parties or other sources (e.g., websites)?**
    
    Other sources: collected by authors.


8. **Were the individuals in question notified about the data collection?** *(If so, please describe (or show with screenshots or other information) how notice was provided, and provide a link or other access point to, or otherwise reproduce, the exact language of the notification itself.)*
    
    No, they were not notified.


9. **Did the individuals in question consent to the collection and use of their data?** *(If so, please describe (or show with screenshots or other information) how consent was requested and provided, and provide a link or other access point to, or otherwise reproduce, the exact language to which the individuals consented.)*
    
    No. 


10. **If consent was obtained, were the consenting individuals provided with a mechanism to revoke their consent in the future or for certain uses?** *(If so, please provide a description, as well as a link or other access point to the mechanism (if appropriate).)*
    
    N/A.


11. **Has an analysis of the potential impact of the dataset and its use on data subjects (e.g., a data protection impact analysis) been conducted?** *(If so, please provide a description of this analysis, including the outcomes, as well as a link or other access point to any supporting documentation.)*
    
    Yes. Our dataset only used for research and without any individual information. Everyone must apply for the authority of this dataset.


12. **Any other comments?**
    
    None.





## Preprocessing/cleaning/labeling


1. **Was any preprocessing/cleaning/labeling of the data done (e.g., discretization or bucketing, tokenization, part-of-speech tagging, SIFT feature extraction, removal of instances, processing of missing values)?** *(If so, please provide a description. If not, you may skip the remainder of the questions in this section.)*
    
    Yes. We utilize the pre-trained single-view tracker to initialize the object bounding boxes and tracklets, which can significantly reduce the labor cost of annotation.


2. **Was the "raw" data saved in addition to the preprocessed/cleaned/labeled data (e.g., to support unanticipated future uses)?** *(If so, please provide a link or other access point to the "raw" data.)*
    
    No.


3. **Is the software used to preprocess/clean/label the instances available?** *(If so, please provide a link or other access point.)*
    
    Yes; it is [CenterNet](https://github.com/xingyizhou/CenterNet) and [VIA-VGG Image Annotator](https://gitlab.com/vgg/via).


4. **Any other comments?**
    
    None.





## Uses


1. **Has the dataset been used for any tasks already?** *(If so, please provide a description.)*
    
    The dataset has been used in cross-view multi-object tracking, and we mention that in the paper.


2. **Is there a repository that links to any or all papers or systems that use the dataset?** *(If so, please provide a link or other access point.)*
    
    No.


3. **What (other) tasks could the dataset be used for?**
    
    The dataset could possibly be used for object detection. 


4. **Is there anything about the composition of the dataset or the way it was collected and preprocessed/cleaned/labeled that might impact future uses?** *(For example, is there anything that a future user might need to know to avoid uses that could result in unfair treatment of individuals or groups (e.g., stereotyping, quality of service issues) or other undesirable harms (e.g., financial harms, legal risks)  If so, please provide a description. Is there anything a future user could do to mitigate these undesirable harms?)*
    
    No.

5. **Are there tasks for which the dataset should not be used?** *(If so, please provide a description.)*
    
    This dataset should not be used for any sort of "gender prediction." First, anyone using this dataset (or any related dataset, for that matter), should recognize that "gender" doesn't mean any single thing, and furthermore that pronoun != gender. Furthermore, because of the fluid and temporal notion of gender--and of gendered referring expressions like pronouns and terms of address--just because a person is described in this dataset in one particular way, does not mean that this will always be the appropriate way to refer to this person.


6. **Any other comments?**
    
    None.




## Distribution


1. **Will the dataset be distributed to third parties outside of the entity (e.g., company, institution, organization) on behalf of which the dataset was created?** *(If so, please provide a description.)*
    
    Yes, the dataset is freely available.


2. **How will the dataset will be distributed (e.g., tarball  on website, API, GitHub)?** *(Does the dataset have a digital object identifier (DOI)?)*
    
    The dataset is free for download at ```https://github.com/shengyuhao/DIVOTrack``` after the user applies for authority.


3. **When will the dataset be distributed?**
    
    The dataset is distributed as of June 2022 in its first version.


4. **Will the dataset be distributed under a copyright or other intellectual property (IP) license, and/or under applicable terms of use (ToU)?** *(If so, please describe this license and/or ToU, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms or ToU, as well as any fees associated with these restrictions.)*
    
    Please see the [LICENSE](https://github.com/shengyuhao/DIVOTrack/blob/main/LICENSE.md).


5. **Have any third parties imposed IP-based or other restrictions on the data associated with the instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any relevant licensing terms, as well as any fees associated with these restrictions.)*
    
    No.


6. **Do any export controls or other regulatory restrictions apply to the dataset or to individual instances?** *(If so, please describe these restrictions, and provide a link or other access point to, or otherwise reproduce, any supporting documentation.)*
    
    Not to our knowledge.


7. **Any other comments?**
    
    None.





## Maintenance


1. **Who is supporting/hosting/maintaining the dataset?**
    
    All authors are maintaining. Shengyu Hao and Peiyuan Liu are hosting on github.


2. **How can the owner/curator/manager of the dataset be contacted (e.g., email address)?**
    
    E-mail addresses are at the bottom on the github.


3. **Is there an erratum?** *(If so, please provide a link or other access point.)*
    
    Currently, no. As errors are encountered, future versions of the dataset may be released (but will be versioned). They will all be provided in the same github location.


4. **Will the dataset be updated (e.g., to correct labeling errors, add new instances, delete instances')?** *(If so, please describe how often, by whom, and how updates will be communicated to users (e.g., mailing list, GitHub)?)*
    
    Same as previous.


5. **If the dataset relates to people, are there applicable limits on the retention of the data associated with the instances (e.g., were individuals in question told that their data would be retained for a fixed period of time and then deleted)?** *(If so, please describe these limits and explain how they will be enforced.)*
    
    Yes. Any person in the dataset has authority to delete the video or relevant images.


6.  **Will older versions of the dataset continue to be supported/hosted/maintained?** *(If so, please describe how. If not, please describe how its obsolescence will be communicated to users.)*
    
    Yes; all data will be versioned.


7.  **If others want to extend/augment/build on/contribute to the dataset, is there a mechanism for them to do so?** *(If so, please provide a description. Will these contributions be validated/verified? If so, please describe how. If not, why not? Is there a process for communicating/distributing these contributions to other users? If so, please provide a description.)*
    
    Errors may be submitted via the bugtracker on github. More extensive augmentations may be accepted at the authors' discretion.


8.  **Any other comments?**
    
    None.


