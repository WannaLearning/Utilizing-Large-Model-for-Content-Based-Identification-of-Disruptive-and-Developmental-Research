# Utilizing Large Model for Content-Based Identification of Disruptive and Developmental Research  
This study not only provides an effective method for timely identifying disruptive research, but also provides guidance for promoting fair scientific evaluation.  

**(1)** We proposed a simple but effective data construction method, which can automatically build a three-class dataset (disruptive, developmental, and consolidating) based on the disruption index (DI).  
**(2)** We employed a range of LLMs with adapter to train a text classifier which can effectively distinguish potentially disruptive and developmental research from general reserch.  
**(3)** Further experimental results on two evaluation datasets show that our model can differentiate between Nobel Prize papers and randomly matched papers, as well as between newly published papers in journals of varying impact factors (IF).   
**(4)** Our findings also show that evaluating papers based solely on IF is insufficient, and a paper published in a general journal boasting high view counts tend to be disruptive or developmental research. 

## Nobel evaluation dataset  
We proposed the Nobel Prize paper evaluation dataset, which includes 80 Nobel Prize papers , 76 other papers authored by Nobel laureates, and 209 random papers. 

<p align="center">
  <img src="https://github.com/WannaLearning/Identifying-disruptive-and-developmental-research-based-purely-on-research-content/blob/main/Figures-git/nobel%20evaluation%20dataset.png" width="60%" alt="Construction process of the Nobel Prize paper evaluation dataset"/>
</p>
<div align="center">
  Figure 2 Construction process of the Nobel Prize paper evaluation dataset
</div>  

## The newly published paper evaluation dataset  
We collected 170 recently published papers, most of which were published in 2023 or 2024, to form the newly published paper evaluation dataset. As shown in Figure 3, we randomly selected newly published papers from the official homepages of distinct biomedicine journals. Specifically, we collected 22 papers from JAMA, BMJ Open, Journal of Clinical Medicine (JCM), American Journal of Epidemiology (AJE), and Medical Science Monitor (MSM), respectively. We also gathered 22 papers from Cell and its sub-journals, along with 38 papers from The Lancet and its sub-journals. JAMA, Cell, and The Lancet are top journals with higher IF, while the others are general journals with lower IF.

<p align="center">
  <img src="https://github.com/WannaLearning/Identifying-disruptive-and-developmental-research-based-purely-on-research-content/blob/main/Figures-git/newly%20published%20papers%20evaluation%20dataset.png" width="45%" alt="Construction process of the newly published paper evaluation dataset"/>
</p>
<div align="center">
  Figure 3 Construction process of the newly published paper evaluation dataset
</div>  

## The titles of the newly published papers
**Table 8. Results on papers published in *JAMA*(https://jamanetwork.com/)**  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |Anxiety and Depression Symptoms After the Dobbs Abortion Decision |Disruptive |
| 2  |Changes in Health Care Workers’ Economic Outcomes Following Medicaid Expansion |General |
| 3  |Cancer Diagnoses After Recent Weight Loss |Disruptive |
| 4  |Functional Outcomes After Localized Prostate Cancer Treatment |Disruptive |
| 5  |HIV Preexposure Prophylaxis With Emtricitabine and Tenofovir Disoproxil Fumarate Among Cisgender Women  |Disruptive |
| 6  |Long-Term Outcomes of Medical Management vs Bariatric Surgery in Type 2 Diabetes  |Disruptive |
| 7  |Time to Treatment With Intravenous Thrombolysis Before Thrombectomy and Functional Outcomes in Acute Ischemic StrokeA Meta-Analysis |Disruptive|
| 8  |Screening for Speech and Language Delay and Disorders in Children  |General |
| 9  |Screening and Preventive Interventions for Oral Health in Adults |Disruptive |
| 10 |Screening and Preventive Interventions for Oral Health in Children and Adolescents Aged 5 to 17 Years |General|
| 11 |Apixaban to Prevent Recurrence After Cryptogenic Stroke in Patients With Atrial Cardiopathy |Developmental|
| 12 |Continued Treatment With Tirzepatide for Maintenance of Weight Reduction in Adults With ObesityThe SURMOUNT-4 Randomized Clinical Trial |Developmental|
| 13 |Effect of Higher-Dose Ivermectin for 6 Days vs Placebo on Time to Sustained Recovery in Outpatients With COVID-19A Randomized Clinical Trial |Disruptive|
| 14 |International Consensus Criteria for Pediatric Sepsis and Septic Shock |Developmental |
| 15 |Neonatal Outcomes After COVID-19 Vaccination in Pregnancy |Disruptive|
| 16 |RNA Interference With Zilebesiran for Mild to Moderate HypertensionThe KARDIA-1 Randomized Clinical Trial |Disruptive|
| 17 |Characteristics and Outcomes of US Children and Adolescents With Multisystem Inflammatory Syndrome in Children (MIS-C) Compared With Severe Acute COVID-19 |Disruptive|
| 18 |Effect of 2 Inactivated SARS-CoV-2 Vaccines on Symptomatic COVID-19 Infection in AdultsA Randomized Clinical Trial |Disruptive|
| 19 |Pancreatic Cancer A Review |Disruptive|
| 20 |Screening for Colorectal CancerUS Preventive Services Task Force Recommendation Statement |Disruptive|
| 21 |Screening for Lung Cancer |Disruptive|
| 22 |Strengthening the Reporting of Observational Studies in Epidemiology Using Mendelian RandomizationThe STROBE-MR Statement |Disruptive|

**Table 9 Results on papers published in *The Lancet*(https://www.thelancet.com/)**  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |A 14-gene B-cell immune signature in early-stage triple-negative breast cancer (TNBC): a pooled analysis of seven studies |Disruptive |
| 2  |AAV1-hOTOF gene therapy for autosomal recessive deafness 9: a single-arm trial |Disruptive |
| 3  |Association between early life exposure to agriculture, biodiversity, and green space and risk of inflammatory bowel disease: a population-based cohort study|Developmental |
| 4  |Cemiplimab in locally advanced or metastatic cutaneous squamous cell carcinoma: prospective real-world data from the DRUG Access Protocol|Developmental |
| 5  |Differential anti-viral response to respiratory syncytial virus A in preterm and term infants |General |
| 6  |Estimates of hospitalisations and deaths in patients with COVID-19 associated with undiagnosed diabetes during the first phase of the pandemic in eight low-income and middle-income countries: a modelling study|General |
| 7  |Efficacy of typhoid conjugate vaccine: final analysis of a 4-year, phase 3, randomised controlled trial in Malawian children |Disruptive|
| 8  |Evaluating the efficacy and safety of pozelimab in patients with CD55 deficiency with hyperactivation of complement, angiopathic thrombosis, and protein-losing enteropathy disease: an open-label phase 2 and 3 study|Developmental |
| 9  |Individualised neoantigen therapy mRNA-4157 (V940) plus pembrolizumab versus pembrolizumab monotherapy in resected melanoma (KEYNOTE-942): a randomised, phase 2b study|Developmental |
| 10 |Indomethacin with or without prophylactic pancreatic stent placement to prevent pancreatitis after ERCP: a randomised non-inferiority trial|Developmental|
| 11 |Term planned delivery based on fetal growth assessment with or without the cerebroplacental ratio in low-risk pregnancies (RATIO37): an international, multicentre, open-label, randomised controlled trial |General|
| 12 |The effect of computerised decision support alerts tailored to intensive care on the administration of high-risk drug combinations, and their monitoring: a cluster randomised stepped-wedge trial |Developmental|
| 13 |Undervaccination and severe COVID-19 outcomes: meta-analysis of national cohort studies in England, Northern Ireland, Scotland, and Wales |Disruptive|
| 14 |Early childhood appetitive traits and eating disorder symptoms in adolescence: a 10-year longitudinal follow-up study in the Netherlands and the UK |General |
| 15 |Effect of parental touch on relieving acute procedural pain in neonates and parental anxiety (Petal): a multicentre, randomised controlled trial in the UK |Developmental|
| 16 |Immunomodulatory therapy in children with paediatric inflammatory multisystem syndrome temporally associated with SARS-CoV-2 (PIMS-TS, MIS-C; RECOVERY): a randomised, controlled, open-label, platform trial |Disruptive|
| 17 |Post-mortem investigation of deaths due to pneumonia in children aged 1–59 months in sub-Saharan Africa and South Asia from 2016 to 2022: an observational study |Disruptive|
| 18 |Long-term outcomes after severe childhood malnutrition in adolescents in Malawi (LOSCM): a prospective observational cohort study |Developmental|
| 19 |Duration of fracture prevention after zoledronate treatment in women with osteopenia: observational follow-up of a 6-year randomised controlled trial to 10 years |Developmental|
| 20 |Efficacy and safety of once weekly semaglutide 2·4 mg for weight management in a predominantly east Asian population with overweight or obesity (STEP 7): a double-blind, multicentre, randomised controlled trial |Disruptive|
| 21 |Glucagon and GLP-1 receptor dual agonist survodutide for obesity: a randomised, double-blind, placebo-controlled, dose-finding phase 2 trial |Developmental|
| 22 |PCSK9 inhibition with orally administered NNC0385-0434 in hypercholesterolaemia: a randomised, double-blind, placebo-controlled and active-controlled phase 2 trial |Disruptive|

**Table 10 Results on papers published in *Cell*(https://www.cell.com/)**  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |DNA-guided transcription factor cooperativity shapes face and limb mesenchyme|Developmental |
| 2  |Cell surface RNAs control neutrophil recruitment |Developmental |
| 3  |Time resolution in cryo-EM using a PDMS-based microfluidic chip assembly and its application to the study of HflX-mediated ribosome recycling|Developmental |
| 4  |Dual phosphorylation of DGK5-mediated PA burst regulates ROS in plant immunity|Developmental |
| 5  |Inherited blood cancer predisposition through altered transcription elongation |Disruptive |
| 6  |Hypoxia and intra-complex genetic suppressors rescue complex I mutants by a shared mechanism |Developmental |
| 7  |Therapeutic application of human type 2 innate lymphoid cells via induction of granzyme B-mediated tumor cell death |Developmental|
| 8  |Immune evasion, infectivity, and fusogenicity of SARS-CoV-2 BA.2.86 and FLip variants |Developmental |
| 9  |SARS-CoV-2 BA.2.86 enters lung cells and evades neutralizing antibodies with high efficiency |Disruptive |
| 10 |Human fetal brain self-organizes into long-term expanding organoids |Disruptive|
| 11 |Neutrophil profiling illuminates anti-tumor antigen-presenting potency |Disruptive|
| 12 |A cryptic plasmid is among the most numerous genetic elements in the human gut |Disruptive|
| 13 |Structure-based design of non-hypertrophic apelin receptor modulator |Developmental|
| 14 |Underdetected dispersal and extensive local transmission drove the 2022 mpox epidemic |Disruptive |
| 15 |Maternal inflammation regulates fetal emergency myelopoiesis |Disruptive|
| 16 |CRB1-associated retinal degeneration is dependent on bacterial translocation from the gut |Developmental|
| 17 |Learning attentional templates for value-based decision-making |Disruptive|
| 18 |A multivalent mRNA monkeypox virus vaccine (BNT166) protects mice and macaques from orthopoxvirus disease |Disruptive|
| 19 |Xist ribonucleoproteins promote female sex-biased autoimmunity |Developmental|
| 20 |A retroviral link to vertebrate myelination through retrotransposon-RNA-mediated control of myelin gene expression |Disruptive|
| 21 |Inosine induces stemness features in CAR-T cells and enhances potency |Disruptive|
| 22 |A comprehensive clinically informed map of dependencies in cancer cells and framework for target prioritization |Disruptive|
| 23 |Interferon-stimulated neutrophils as a predictor of immunotherapy response |Disruptive|
| 24 |Multi-omic profiling of follicular lymphoma reveals changes in tissue architecture and enhanced stromal remodeling in high-risk patients |Disruptive|
| 25 |PDGFRα+ITGA11+ fibroblasts foster early-stage cancer lymphovascular invasion and lymphatic metastasis via ITGA11-SELE interplay |Disruptive|
| 26 |Retinoic acid receptor activation reprograms senescence response and enhances anti-tumor activity of natural killer cells |Disruptive|
| 27 |A living biobank of patient-derived ductal carcinoma in situ mouse-intraductal xenografts identifies risk factors for invasive progression |Developmental|
| 28 |Interferon-stimulated neutrophils as a predictor of immunotherapy response |Disruptive|
| 29 |A comprehensive clinically informed map of dependencies in cancer cells and framework for target prioritization |Disruptive|
| 30 |Antiplasmodial peptaibols act through membrane directed mechanisms |Developmental|
| 31 |PIM1 targeted degradation prevents the emergence of chemoresistance in prostate cancer |Developmental|
| 32 |A sterol analog inhibits hedgehog pathway by blocking cholesterylation of smoothened |Developmental|
| 33 |Allosteric inhibition of tRNA synthetase Gln4 by N-pyrimidinyl-β-thiophenylacrylamides exerts highly selective antifungal activity |Developmental|
| 34 |CYP7B1-mediated 25-hydroxycholesterol degradation maintains quiescence-activation balance and improves therapeutic potential of mesenchymal stem cells |General|
| 35 |Identification of differential biological activity and synergy between the PARP inhibitor rucaparib and its major metabolite |Disruptive|
| 36 |Small molecule targeting of transcription-replication conflict for selective chemotherapy |Developmental|
| 37 |The cyclimids: Degron-inspired cereblon binders for targeted protein degradation |Developmental|
| 38 |PIM1 targeted degradation prevents the emergence of chemoresistance in prostate cancer |Developmental|


**Table 11 Results on papers published in *BMJ Open*(https://bmjopen.bmj.com/)**  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |Association between leisure sedentary behaviour and uterine fibroids in non-menopausal women: a population-based study |General |
| 2  |Associations of smoking and alcohol consumption with the development of open angle glaucoma: a retrospective cohort study |General |
| 3  |Association between kimchi consumption and obesity based on BMI and abdominal obesity in Korean adults: a cross-sectional analysis of the Health Examinees study |Developmental |
| 4  |Assessment of quality of data submitted for NICE technology appraisals over two decades |General |
| 5  |Collaborative care model versus usual care for the management of musculoskeletal and co-existing mental health conditions: a randomised feasibility mixed-methods study |General |
| 6  |Development and evaluation of a manualised mental health awareness and stigma reduction intervention for Black faith communities: study protocol for the ON TRAC feasibility study |General |
| 7  |Evaluating a stepped care model of psychological support for adults affected by adversity: study protocol for a randomised controlled trial in Jordan |General|
| 8  |Experience-based Investigation and Co-design of Psychosis Centred Integrated Care Services for Ethnically Diverse People with Multimorbidity (CoPICS): study protocol |General |
| 9  |Effect of acupuncture on ischaemic stroke in patients with rheumatoid arthritis: a nationwide propensity score-matched study |General |
| 10 |Energy drink consumption and sleep parameters in college and university students: a national cross-sectional study|Developmental|
| 11 |Is male gynaecomastia associated with an increased risk of death? A nationwide register-based cohort study |Developmental|
| 12 |Is the awarding gap at UK medical schools influenced by ethnicity and medical school attended? A retrospective cohort study|Developmental|
| 13 |Towards integrated mental health services in low-income and middle-income countries: organisation of primary healthcare providers – a scoping review protocol |General|
| 14 |Use of drugs for hyperlipidaemia and diabetes and risk of primary and secondary brain tumours: nested case–control studies using the UK Clinical Practice Research Datalink (CPRD) |General |
| 15 |Career intentions of medical students in the UK: a national, cross-sectional study (AIMS study) |Disruptive|
| 16 |Does the advertisement in Swiss pharmacy windows rest on evidence-based medicine? An observational study |General|
| 17 |Effect of multi-level interventions on mental health outcomes among adolescents in sub-Saharan Africa: a systematic review |General|
| 18 |Global, regional and national burden of inflammatory bowel disease in 204 countries and territories from 1990 to 2019: a systematic analysis based on the Global Burden of Disease Study 2019 |Disruptive|
| 19 |Impact of Long Covid on the school experiences of children and young people: a qualitative study |Developmental|
| 20 |Risk of myocarditis and pericarditis in mRNA COVID-19-vaccinated and unvaccinated populations: a systematic review and meta-analysis |Disruptive|
| 21 |Silver linings of ADHD: a thematic analysis of adults’ positive experiences with living with ADHD |General|
| 22 |What are the treatment remission, response and extent of improvement rates after up to four trials of antidepressant therapies in real-world depressed patients? |Disruptive|


**Table 12 Results on papers published in *Journal of Clinical Medicine*(https://www.mdpi.com/journal/jcm)**  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |A Non-Coronary, Peripheral Arterial Atherosclerotic Disease (Carotid, Renal, Lower Limb) in Elderly Patients—A Review PART II—Pharmacological Approach for Management of Elderly Patients with Peripheral Atherosclerotic Lesions outside Coronary Territory |General |
| 2  |Bone Remodeling of Maxilla after Retraction of Incisors during Orthodontic Treatment with Extraction of Premolars Based on CBCT Study: A Systematic Review |General |
| 3  |Congenital Optic Disc Anomalies: Insights from Multimodal Imaging |Developmental |
| 4  |Comparison of Quality of Recovery between Modified Thoracoabdominal Nerves Block through Perichondrial Approach versus Oblique Subcostal Transversus Abdominis Plane Block in Patients Undergoing Total Laparoscopic Hysterectomy: A Pilot Randomized Controlled Trial |Developmental |
| 5  |Delayed Diagnosis of Spinal Dural Arteriovenous Fistula: A Case Report and Scoping Review |Developmental |
| 6  |Efficacy and Safety of Low-Dose Atropine on Myopia Prevention in Premyopic Children: Systematic Review and Meta-Analysis |Developmental |
| 7  |Glioblastoma and Internal Carotid Artery Calcium Score: A Possible Novel Prognostic Partnership? |General|
| 8  |How Reliable Is Breast Volume Assessment When the Patient Is Lying Flat?—Volumetric Assessment of Breast Volume Using a Vectra H2 Handheld Device in Different Positions |Developmental |
| 9  |How to Manage Advanced Differentiated Thyroid Cancer: Step-by-Step Analysis from Two Italian Tertiary Referral Centers |General |
| 10 |Integra® Dermal Regeneration Template in Complex Scalp Reconstruction |General|
| 11 |Oral Manifestations in Pregnant Women: A Systematic Review |General|
| 12 |Prolongated Activated Partial Thromboplastin Time (aPTT) in Pediatric Patients before Surgery—Crying Wolf: Lupus (Anticoagulant) Does Not Always Threaten Children |General|
| 13 |The Diagnostic Accuracy of Transcranial Color-Coded Doppler Ultrasound Technique in Stratifying Intracranial Cerebral Artery Stenoses in Cerebrovascular Disease Patients: A Systematic Review and Meta-Analysis  |General|
| 14 |The Association of High-Molecular-Weight Hyaluronic Acid (HMWHA), Alpha Lipoic Acid (ALA), Magnesium, Vitamin B6, and Vitamin D Improves Subchorionic Hematoma Resorption in Women with Threatened Miscarriage: A Pilot Clinical Study|Developmental |
| 15 |Using Artificial Intelligence to Predict Mechanical Ventilation Weaning Success in Patients with Respiratory Failure, Including Those with Acute Respiratory Distress Syndrome |General|
| 16 |Using Augmented Reality Technology to Optimize Transfacet Lumbar Interbody Fusion: A Case Report |General|
| 17 |Uncommon Blepharitis |General|
| 18 |Albuminuria, Forgotten No More: Underlining the Emerging Role in CardioRenal Crosstalk |General|
| 19 |Are Surgeons Going to Be Left Holding the Bag? Incisional Hernia Repair and Intra-Peritoneal Non-Absorbable Mesh Implant Complications |General|
| 20 |Role of Female Sex Hormones in ADPKD Progression and a Personalized Approach to Contraception and Hormonal Therapy |General|
| 21 |Recent Advances and Future Directions in Syncope Management: A Comprehensive Narrative Review |General|
| 22 |Long-COVID Prevalence and Its Association with Health Outcomes in the Post-Vaccine and Antiviral-Availability Era |General|


**Table 13 Results on papers published in *American Journal of Epidemiology*(https://academic.oup.com/aje)**  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |Missing data and missed infections: Investigating racial and ethnic disparities in SARS-CoV-2 testing and infection rates in Holyoke, Massachusetts |General |
| 2  |Machine Learning Detects Heterogeneous Effects of Medicaid Coverage on Depression |Disruptive |
| 3  |Handling missing data when estimating causal effects with Targeted Maximum Likelihood Estimation |Developmental |
| 4  |The InterSECT framework: A proposed model for explaining population-level trends in substance use and emotional concerns |General |
| 5  |Understanding Racial/Ethinic Disparities in COVID Mortality Using a Novel Metric: COVID Excess Mortality Percentage |Disruptive |
| 6  |Standardizing to Specific Target Populations in Distributed Networks and Multi-Site Pharmacoepidemiologic Studies |General |
| 7  |Measuring Variation in Infant Mortality and Deaths of Despair by U.S. Congressional Districts in Pennsylvania: A Methodological Case Study |General|
| 8  |Estimating effects of longitudinal and cumulative exposure to PFAS mixtures on early adolescent body composition |Developmental |
| 9  |An Introduction to Bayesian Spatial Smoothing Methods for Disease Mapping: Modeling County Firearm Suicide Mortality Rates |General |
| 10 |Omics feature selection with the extended SIS R package: identification of a body mass index epigenetic multi-marker in the Strong Heart Study |General|
| 11 |Month-to-month all-cause mortality forecasting: A method allowing for changes in seasonal patterns |General|
| 12 |Exercise to socialize? Bidirectional relationships between physical activity and loneliness in middle-aged and older American adults |General|
| 13 |Harnessing Causal Forests for Epidemiologic Research: Key Consideration |Developmental|
| 14 |The World Was Their Laboratory: How Two Pioneer Scientist-Administrators,James Watt and Zdenek Fejfar, Advanced Methods and International Collaboration in Cardiovascular Disease Epidemiology During the Cold War |General |
| 15 |Life Expectancy among Native Americans during the COVID-19 Pandemic: Estimates, Uncertainty and Obstacles |General|
| 16 |Estimating protection afforded by prior infection in preventing reinfection: Applying the test-negative study design |Disruptive|
| 17 |The application of target trials with longitudinal targeted maximum likelihood estimation to assess the effect of alcohol consumption in adolescence on depressive symptoms in adulthood |General|
| 18 |Associations of Accelerometer-Measured and Self-Reported Sedentary Time With Leukocyte Telomere Length in Older Women |Developmental|
| 19 |Young Age at First Sexual Intercourse and Sexually Transmitted Infections in Adolescents and Young Adults |Disruptive|
| 20 |The Relationship Between Occupational Standing and Sitting and Incident Heart Disease Over a 12-Year Period in Ontario, Canada |Developmental|
| 21 |Death and Chronic Disease Risk Associated With Poor Life Satisfaction: A Population-Based Cohort Study |General|
| 22 |What is Machine Learning? A Primer for the Epidemiologist |Disruptive|

**Table 14 Results on papers published in *Medical Science Monitor(https://medscimonit.com/)***  
| Id| Title  |Type|
| ------------- | ------------- | ------------- |
| 1  |Bibliometric Analysis of Brain Stimulation Technologies in Sleep Disorders |General |
| 2  |Clinical Outcomes of Arthroscopic Surgery in Patients with Gluteal Muscle Contracture: Single-Institution Results from a High-Volume Cohort |Developmental |
| 3  |Comparison of Modified Shock Index and Shock Index for Predicting Massive Transfusion in Women with Primary Postpartum Hemorrhage: A Retrospective Study |Developmental |
| 4  |Diagnostic Accuracy of Ileocolic Artery and Vein Diameter for Acute Appendicitis |General |
| 5  |Enhancing Patient Adherence to Newly-Prescribed Medicine for Chronic Diseases: A Comprehensive Review and Cost-Effective Approach to Implementing the New Medicine Service in Community Pharmacies in Poland |General |
| 6  |Efficacy of Bakri Intrauterine Balloon in Managing Postpartum Hemorrhage: A Comparative Analysis of Vaginal and Cesarean Deliveries with Placenta Accreta Spectrum Disorders |Disruptive |
| 7  |Early Mobilization in Pediatric Critical Care: Exploring the Gap Between Theory and Practice in Saudi Arabia |Developmental|
| 8  |Ecoepidemiology of Ancylostoma spp. in Urban-Marginal and Rural Sectors of the Ecuadorian Coast and Prevalence of Cutaneous Larvae Migrans |General |
| 9  |Impact of Auxiliary Features on Retention of Short Dental Crowns: An In-Vitro Analysis of Box and Groove Preparations |General |
| 10 |Patterns and Outcomes of Traumatic Suicides: A Retrospective Study of 132 Patients Admitted to a Turkish Medical Center |General|
| 11 |Review of the Evaluation of Pulmonary Hypoplasia as an Important Determinant of Clinical Outcomes in Infants with Congenital Diaphragmatic Hernia |General|
| 12 |Predictors of Mortality in Veterans with Amyotrophic Lateral Sclerosis: Respiratory Status and Speech Disorder at Presentation |General|
| 13 |Molecular Determinants of Drug Resistance and Mutation Patterns in Influenza Viruses Circulating in Poland Across Multiple Epidemic Seasons: Implications for Vaccination Strategies |General|
| 14 |Electrophysiological Testing for an Auditory Processing Disorder and Reading Performance in 54 School Students Aged Between 8 and 12 years |General |
| 15 |Vaccination Guidelines for Pregnant Women: Addressing COVID-19 and the Omicron Variant |General|
| 16 |Prevalence and Variability of Allergen-Specific Immunoglobulin E in Patients with Elevated Tryptase Levels |General|
| 17 |Association Between Sleep Duration and Sleep Disorder Data from the National Health and Nutrition Examination Survey and Stroke Among Adults in the United States |General|
| 18 |A Review of the Potential Roles of Antioxidant and Anti-Inflammatory Pharmacological Approaches for the Management of Mild-to-Moderate Symptomatic COVID-19 |Disruptive|
| 19 |Effect of Physiotherapy to Correct Rounded Shoulder Posture in 30 Patients During the COVID-19 Pandemic in South Korea Using a Telerehabilitation Exercise Program to Improve Posture, Physical Function, and Reduced Pain, with Evaluation of Patient Satisfaction |General|
| 20 |Efficacy of Focused Extracorporeal Shock Wave Therapy in Chronic Low Back Pain: A Prospective Randomized 3-Month Follow-Up Study |General|
| 21 |Virtual Screening and Molecular Docking to Study the Mechanism of Chinese Medicines in the Treatment of Coronavirus Infection |General|
| 22 |Questionnaire-Based Study on the Prevalence, Awareness, and Preventive Measures of Occupational Hazards Among Dental Professionals |General|
