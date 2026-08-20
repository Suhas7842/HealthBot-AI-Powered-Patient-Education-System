"""
Medical test suite for evaluating HealthBot's RAG and generation quality.
Contains 50 carefully curated medical questions with ground truth answers.

For retrieval evaluation, relevant document IDs are generated based on
condition matching from the knowledge base.
"""

from typing import Optional

# Medical test cases covering common conditions
MEDICAL_TEST_CASES: list[dict[str, str]] = [
    # Diabetes (5 cases)
    {
        "question": "What are the main symptoms of Type 2 diabetes?",
        "ground_truth": "Increased thirst, frequent urination, increased hunger, fatigue, blurred vision, slow-healing sores, frequent infections, numbness or tingling in hands or feet",
        "condition": "diabetes",
    },
    {
        "question": "What causes Type 1 diabetes?",
        "ground_truth": "Type 1 diabetes is caused by an autoimmune reaction where the body's immune system attacks and destroys insulin-producing beta cells in the pancreas",
        "condition": "diabetes",
    },
    {
        "question": "How is diabetes diagnosed?",
        "ground_truth": "Diabetes is diagnosed through blood tests including fasting plasma glucose test (≥126 mg/dL), HbA1c test (≥6.5%), oral glucose tolerance test (≥200 mg/dL), or random plasma glucose test (≥200 mg/dL with symptoms)",
        "condition": "diabetes",
    },
    {
        "question": "What are the risk factors for Type 2 diabetes?",
        "ground_truth": "Risk factors include obesity, physical inactivity, family history, age over 45, gestational diabetes history, polycystic ovary syndrome, high blood pressure, abnormal cholesterol levels, and certain ethnicities",
        "condition": "diabetes",
    },
    {
        "question": "What lifestyle changes help manage Type 2 diabetes?",
        "ground_truth": "Healthy eating, regular physical activity, weight loss, blood sugar monitoring, diabetes medications or insulin therapy, and stress management",
        "condition": "diabetes",
    },
    # Hypertension (5 cases)
    {
        "question": "What is considered high blood pressure?",
        "ground_truth": "High blood pressure (hypertension) is defined as systolic pressure of 130 mmHg or higher, or diastolic pressure of 80 mmHg or higher",
        "condition": "hypertension",
    },
    {
        "question": "What are the symptoms of hypertension?",
        "ground_truth": "Hypertension is often called the silent killer because it typically has no symptoms. Severe hypertension may cause headaches, shortness of breath, nosebleeds, chest pain, visual changes, or blood in urine",
        "condition": "hypertension",
    },
    {
        "question": "What causes high blood pressure?",
        "ground_truth": "Causes include unhealthy lifestyle (high sodium diet, lack of physical activity, obesity), age, family history, chronic conditions (kidney disease, diabetes), stress, smoking, and excessive alcohol consumption",
        "condition": "hypertension",
    },
    {
        "question": "How can hypertension be prevented?",
        "ground_truth": "Prevention includes maintaining healthy weight, regular exercise, reducing sodium intake, limiting alcohol, not smoking, managing stress, eating a balanced diet rich in fruits and vegetables, and regular blood pressure monitoring",
        "condition": "hypertension",
    },
    {
        "question": "What are complications of untreated hypertension?",
        "ground_truth": "Complications include heart attack, stroke, heart failure, kidney damage, vision loss, sexual dysfunction, peripheral artery disease, and aneurysm",
        "condition": "hypertension",
    },
    # Asthma (5 cases)
    {
        "question": "What are the main symptoms of asthma?",
        "ground_truth": "Shortness of breath, chest tightness or pain, wheezing when exhaling, coughing or wheezing attacks worsened by respiratory virus, difficulty sleeping due to breathing problems",
        "condition": "asthma",
    },
    {
        "question": "What triggers asthma attacks?",
        "ground_truth": "Common triggers include airborne allergens (pollen, dust mites, pet dander), respiratory infections, cold air, air pollutants, physical activity, stress, certain medications (beta blockers, aspirin), and gastroesophageal reflux disease",
        "condition": "asthma",
    },
    {
        "question": "How is asthma diagnosed?",
        "ground_truth": "Diagnosis involves medical history, physical examination, lung function tests (spirometry, peak flow test), and sometimes methacholine challenge test, allergy testing, or imaging tests to rule out other conditions",
        "condition": "asthma",
    },
    {
        "question": "What are long-term asthma control medications?",
        "ground_truth": "Long-term control medications include inhaled corticosteroids, leukotriene modifiers, long-acting beta agonists, combination inhalers, and biologics for severe asthma",
        "condition": "asthma",
    },
    {
        "question": "What is an asthma action plan?",
        "ground_truth": "An asthma action plan is a written document developed with your doctor that outlines daily treatment, how to recognize worsening symptoms, what medications to take during an attack, and when to seek emergency care",
        "condition": "asthma",
    },
    # Heart Disease (5 cases)
    {
        "question": "What are the warning signs of a heart attack?",
        "ground_truth": "Chest pain or discomfort, pain in arms/back/neck/jaw/stomach, shortness of breath, cold sweat, nausea, lightheadedness. Women may experience atypical symptoms like fatigue, indigestion, or anxiety",
        "condition": "heart disease",
    },
    {
        "question": "What is coronary artery disease?",
        "ground_truth": "Coronary artery disease occurs when the major blood vessels supplying the heart become damaged or diseased, usually due to plaque buildup (atherosclerosis), reducing blood flow to the heart muscle",
        "condition": "heart disease",
    },
    {
        "question": "What are risk factors for heart disease?",
        "ground_truth": "Risk factors include age, sex, family history, smoking, high blood pressure, high cholesterol, diabetes, obesity, physical inactivity, unhealthy diet, stress, and poor dental health",
        "condition": "heart disease",
    },
    {
        "question": "How is heart disease prevented?",
        "ground_truth": "Prevention includes not smoking, controlling blood pressure and cholesterol, regular exercise, maintaining healthy weight, eating heart-healthy diet, managing diabetes, reducing stress, and getting adequate sleep",
        "condition": "heart disease",
    },
    {
        "question": "What tests diagnose heart disease?",
        "ground_truth": "Tests include electrocardiogram (ECG), echocardiogram, stress test, cardiac catheterization, coronary angiography, cardiac CT or MRI, and blood tests for cardiac markers",
        "condition": "heart disease",
    },
    # Depression (5 cases)
    {
        "question": "What are the symptoms of major depression?",
        "ground_truth": "Persistent sad or empty mood, loss of interest in activities, significant weight change, sleep disturbances, fatigue, feelings of worthlessness or guilt, difficulty concentrating, recurrent thoughts of death or suicide",
        "condition": "depression",
    },
    {
        "question": "What causes depression?",
        "ground_truth": "Depression results from complex interaction of biological factors (brain chemistry, hormones, genetics), psychological factors (trauma, low self-esteem), and environmental factors (stress, loss, difficult circumstances)",
        "condition": "depression",
    },
    {
        "question": "How is depression treated?",
        "ground_truth": "Treatment includes psychotherapy (cognitive behavioral therapy, interpersonal therapy), antidepressant medications (SSRIs, SNRIs, others), lifestyle changes, support groups, and in severe cases, electroconvulsive therapy or other brain stimulation therapies",
        "condition": "depression",
    },
    {
        "question": "When should someone seek help for depression?",
        "ground_truth": "Seek help if symptoms persist for more than two weeks, interfere with daily functioning, cause significant distress, or include thoughts of self-harm or suicide. Emergency help is needed for suicidal thoughts or plans",
        "condition": "depression",
    },
    {
        "question": "Can depression be prevented?",
        "ground_truth": "While not always preventable, risk can be reduced through stress management, building strong relationships, treating problems early, maintaining healthy lifestyle, getting regular exercise, and avoiding alcohol and drugs",
        "condition": "depression",
    },
    # Arthritis (5 cases)
    {
        "question": "What is the difference between osteoarthritis and rheumatoid arthritis?",
        "ground_truth": "Osteoarthritis is degenerative wear-and-tear arthritis affecting cartilage, common in older adults. Rheumatoid arthritis is an autoimmune disease where the immune system attacks joint linings, can occur at any age and affects multiple joints symmetrically",
        "condition": "arthritis",
    },
    {
        "question": "What are common symptoms of arthritis?",
        "ground_truth": "Joint pain, stiffness (especially in morning or after inactivity), swelling, redness, decreased range of motion, warmth around joints, and in rheumatoid arthritis, fatigue and fever",
        "condition": "arthritis",
    },
    {
        "question": "How is arthritis diagnosed?",
        "ground_truth": "Diagnosis involves medical history, physical examination, blood tests (inflammatory markers, rheumatoid factor, anti-CCP antibodies), imaging tests (X-rays, MRI, ultrasound), and sometimes joint fluid analysis",
        "condition": "arthritis",
    },
    {
        "question": "What treatments are available for arthritis?",
        "ground_truth": "Treatments include pain medications (NSAIDs, acetaminophen), disease-modifying drugs for rheumatoid arthritis, corticosteroids, physical therapy, occupational therapy, lifestyle modifications, and in severe cases, joint surgery",
        "condition": "arthritis",
    },
    {
        "question": "What lifestyle changes help manage arthritis?",
        "ground_truth": "Weight management to reduce joint stress, regular low-impact exercise, hot and cold therapy, joint protection techniques, healthy diet rich in omega-3 fatty acids, adequate rest, and stress management",
        "condition": "arthritis",
    },
    # Migraine (5 cases)
    {
        "question": "What are the phases of a migraine attack?",
        "ground_truth": "Migraine has four phases: prodrome (warning signs 1-2 days before), aura (visual or sensory disturbances), headache (throbbing pain, often one-sided), and postdrome (fatigue and confusion after)",
        "condition": "migraine",
    },
    {
        "question": "What triggers migraines?",
        "ground_truth": "Common triggers include hormonal changes, certain foods and drinks (aged cheese, alcohol, caffeine), stress, sensory stimuli (bright lights, loud sounds), sleep changes, weather changes, medications, and physical exertion",
        "condition": "migraine",
    },
    {
        "question": "How are migraines treated?",
        "ground_truth": "Treatment includes acute medications (triptans, NSAIDs, anti-nausea drugs) taken during attacks, and preventive medications (beta blockers, antidepressants, anti-seizure drugs, CGRP inhibitors) taken daily to reduce frequency",
        "condition": "migraine",
    },
    {
        "question": "What is migraine with aura?",
        "ground_truth": "Migraine with aura involves warning symptoms that occur before or during the headache, including visual disturbances (flashing lights, blind spots), sensory changes (tingling, numbness), or speech difficulties",
        "condition": "migraine",
    },
    {
        "question": "When should someone seek emergency care for a headache?",
        "ground_truth": "Seek emergency care for sudden severe headache, headache with fever/stiff neck/confusion/seizures/numbness/difficulty speaking, headache after head injury, or sudden severe headache different from usual pattern",
        "condition": "migraine",
    },
    # COVID-19 (5 cases)
    {
        "question": "What are the most common symptoms of COVID-19?",
        "ground_truth": "Fever, cough, fatigue, loss of taste or smell, difficulty breathing, body aches, headache, sore throat, congestion, nausea, and diarrhea. Symptoms may appear 2-14 days after exposure",
        "condition": "covid-19",
    },
    {
        "question": "How is COVID-19 transmitted?",
        "ground_truth": "COVID-19 spreads primarily through respiratory droplets when an infected person coughs, sneezes, talks, or breathes. It can also spread through aerosols in enclosed spaces and by touching contaminated surfaces then touching face",
        "condition": "covid-19",
    },
    {
        "question": "Who is at higher risk for severe COVID-19?",
        "ground_truth": "Higher risk groups include older adults (65+), people with chronic conditions (heart disease, diabetes, lung disease, obesity), immunocompromised individuals, pregnant women, and those with certain genetic factors",
        "condition": "covid-19",
    },
    {
        "question": "How can COVID-19 be prevented?",
        "ground_truth": "Prevention includes vaccination, wearing masks in crowded indoor spaces, maintaining physical distance, washing hands frequently, improving ventilation, avoiding crowds, staying home when sick, and testing when exposed or symptomatic",
        "condition": "covid-19",
    },
    {
        "question": "What is long COVID?",
        "ground_truth": "Long COVID (post-COVID conditions) refers to symptoms persisting weeks or months after initial infection, including fatigue, brain fog, shortness of breath, heart palpitations, body aches, and difficulty concentrating",
        "condition": "covid-19",
    },
    # Obesity (5 cases)
    {
        "question": "How is obesity defined medically?",
        "ground_truth": "Obesity is defined by Body Mass Index (BMI): BMI 30-34.9 is Class 1 obesity, 35-39.9 is Class 2, and 40+ is Class 3 (severe obesity). BMI is calculated as weight in kg divided by height in meters squared",
        "condition": "obesity",
    },
    {
        "question": "What health conditions are associated with obesity?",
        "ground_truth": "Associated conditions include Type 2 diabetes, heart disease, stroke, certain cancers, sleep apnea, osteoarthritis, fatty liver disease, kidney disease, pregnancy complications, and mental health issues",
        "condition": "obesity",
    },
    {
        "question": "What causes obesity?",
        "ground_truth": "Obesity results from complex interactions of genetics, metabolism, behavior (diet, physical activity), environment (food availability, built environment), and socioeconomic factors. It occurs when calorie intake exceeds energy expenditure over time",
        "condition": "obesity",
    },
    {
        "question": "What are evidence-based treatments for obesity?",
        "ground_truth": "Treatments include dietary changes, increased physical activity, behavioral therapy, prescription weight-loss medications (GLP-1 agonists, orlistat), and for severe obesity, bariatric surgery (gastric bypass, sleeve gastrectomy)",
        "condition": "obesity",
    },
    {
        "question": "Why is gradual weight loss recommended?",
        "ground_truth": "Gradual weight loss (1-2 pounds per week) is more sustainable, preserves muscle mass, reduces risk of nutritional deficiencies and gallstones, and has better long-term success rates compared to rapid weight loss",
        "condition": "obesity",
    },
    # Stroke (5 cases)
    {
        "question": "What are the warning signs of stroke (FAST)?",
        "ground_truth": "FAST stands for: Face drooping (one side), Arm weakness (one arm drifts down), Speech difficulty (slurred or confused speech), Time to call 911 immediately. Other signs include sudden severe headache, vision problems, dizziness, loss of balance",
        "condition": "stroke",
    },
    {
        "question": "What is the difference between ischemic and hemorrhagic stroke?",
        "ground_truth": "Ischemic stroke (85% of strokes) occurs when a blood clot blocks an artery to the brain. Hemorrhagic stroke occurs when a blood vessel ruptures, causing bleeding in or around the brain",
        "condition": "stroke",
    },
    {
        "question": "What are the risk factors for stroke?",
        "ground_truth": "Risk factors include high blood pressure, heart disease, diabetes, smoking, high cholesterol, obesity, physical inactivity, excessive alcohol, drug use, age, family history, and previous stroke or TIA",
        "condition": "stroke",
    },
    {
        "question": "What is a TIA (transient ischemic attack)?",
        "ground_truth": "A TIA is a mini-stroke with temporary stroke symptoms lasting minutes to hours, caused by brief blood flow interruption. It's a warning sign of potential future stroke and requires immediate medical evaluation",
        "condition": "stroke",
    },
    {
        "question": "Why is immediate treatment critical for stroke?",
        "ground_truth": "Time is brain - brain cells die rapidly without oxygen. Clot-busting medication (tPA) must be given within 4.5 hours of symptom onset for ischemic stroke. Early treatment minimizes brain damage and improves recovery outcomes",
        "condition": "stroke",
    },
]


def get_test_cases_by_condition(condition: str) -> list[dict[str, str]]:
    """
    Get test cases for a specific medical condition.

    Args:
        condition: Medical condition name

    Returns:
        List of test cases for that condition
    """
    return [case for case in MEDICAL_TEST_CASES if case["condition"] == condition]


def get_all_conditions() -> list[str]:
    """
    Get list of all conditions covered in test suite.

    Returns:
        List of unique condition names
    """
    return list(set(case["condition"] for case in MEDICAL_TEST_CASES))


def get_test_suite_stats() -> dict:
    """
    Get statistics about the test suite.

    Returns:
        Dictionary with test suite statistics
    """
    conditions = get_all_conditions()
    return {
        "total_cases": len(MEDICAL_TEST_CASES),
        "num_conditions": len(conditions),
        "conditions": sorted(conditions),
        "cases_per_condition": {
            condition: len(get_test_cases_by_condition(condition))
            for condition in sorted(conditions)
        },
    }


def get_relevant_doc_ids_for_condition(
    condition: str, id_field: str = "chunk_id"
) -> list[str]:
    """
    Get list of relevant document IDs for a given medical condition.

    This generates ground truth for retrieval evaluation by matching
    document chunks with the specified condition.

    Args:
        condition: Medical condition name (e.g., "diabetes", "hypertension")
        id_field: Field to use as document ID ("chunk_id", "pmid")

    Returns:
        List of document IDs (as strings) relevant to the condition

    Example:
        >>> ids = get_relevant_doc_ids_for_condition("diabetes")
        >>> len(ids)
        250  # Approximate number of diabetes-related chunks
    """
    from healthbot.data.processor import DocumentProcessor

    processor = DocumentProcessor()
    chunks = processor.process_knowledge_base()

    # Condition matching - handle variations
    condition_lower = condition.lower()
    condition_variants = {
        "diabetes": ["diabetes", "diabetes mellitus"],
        "hypertension": ["hypertension", "high blood pressure"],
        "heart disease": ["heart disease", "coronary artery disease", "cardiovascular"],
        "asthma": ["asthma"],
        "arthritis": ["arthritis", "osteoarthritis", "rheumatoid arthritis"],
        "depression": ["depression", "depressive disorder"],
        "migraine": ["migraine", "headache"],
        "covid-19": ["covid-19", "sars-cov-2", "coronavirus"],
        "obesity": ["obesity", "overweight"],
        "stroke": ["stroke", "cerebrovascular"],
    }

    # Get matching conditions
    match_conditions = condition_variants.get(
        condition_lower, [condition_lower]
    )

    # Filter chunks by condition
    relevant_ids = []
    for chunk in chunks:
        chunk_condition = chunk.get("condition", "").lower()
        if any(cond in chunk_condition for cond in match_conditions):
            doc_id = str(chunk.get(id_field, ""))
            if doc_id:
                relevant_ids.append(doc_id)

    return relevant_ids


def enrich_test_cases_with_ground_truth(
    test_cases: Optional[list[dict]] = None,
    id_field: str = "chunk_id",
) -> list[dict]:
    """
    Enrich test cases with ground truth relevant document IDs.

    Args:
        test_cases: List of test cases (defaults to MEDICAL_TEST_CASES)
        id_field: Field to use as document ID ("chunk_id", "pmid")

    Returns:
        Test cases enriched with "relevant_doc_ids" field

    Example:
        >>> enriched = enrich_test_cases_with_ground_truth()
        >>> enriched[0]["relevant_doc_ids"][:3]
        ['0', '1', '2']  # First 3 diabetes-related chunk IDs
    """
    if test_cases is None:
        test_cases = MEDICAL_TEST_CASES

    # Cache condition -> doc IDs mapping to avoid recomputing
    condition_cache = {}

    enriched_cases = []
    for case in test_cases:
        enriched_case = case.copy()
        condition = case["condition"]

        # Get relevant doc IDs (cached by condition)
        if condition not in condition_cache:
            condition_cache[condition] = get_relevant_doc_ids_for_condition(
                condition, id_field
            )

        enriched_case["relevant_doc_ids"] = condition_cache[condition]
        enriched_cases.append(enriched_case)

    return enriched_cases


if __name__ == "__main__":
    # Print test suite statistics
    stats = get_test_suite_stats()

    print("=" * 80)
    print("MEDICAL TEST SUITE STATISTICS")
    print("=" * 80)
    print(f"Total test cases: {stats['total_cases']}")
    print(f"Conditions covered: {stats['num_conditions']}")
    print("\nCases per condition:")
    for condition, count in stats["cases_per_condition"].items():
        print(f"  • {condition}: {count} cases")
    print("=" * 80)

    # Show sample test case
    print("\nSample test case:")
    sample = MEDICAL_TEST_CASES[0]
    print(f"Condition: {sample['condition']}")
    print(f"Question: {sample['question']}")
    print(f"Ground truth: {sample['ground_truth'][:100]}...")
    print("=" * 80)
