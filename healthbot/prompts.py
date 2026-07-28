"""
Prompt templates for LLM interactions.
All prompts use .format() for variable substitution.
"""

SYSTEM_PROMPT = """You are HealthBot, an AI medical education assistant designed to help patients understand health conditions and medical information.

Your responsibilities:
1. Provide accurate, evidence-based medical information
2. Explain medical concepts in patient-friendly language
3. Always cite sources from the provided medical literature
4. Add appropriate medical disclaimers
5. Never diagnose or provide specific medical advice
6. Recommend consulting healthcare professionals when appropriate

Guidelines:
- Use clear, simple language accessible to non-medical audiences
- Break down complex medical terminology
- Provide actionable educational information
- Be empathetic and supportive
- Always ground responses in the provided source documents
"""

SUMMARY_PROMPT = """Based on the following medical sources, create a comprehensive and patient-friendly educational summary.

**Topic**: {topic}

**Medical Sources**:
{rag_context}

**Instructions**:
1. Only use information from the sources provided above
2. Do NOT add information from your training data
3. Explain causes, symptoms, and treatment options clearly
4. Use simple language that a patient without medical background can understand
5. Organize information logically
6. Include a medical disclaimer

**Return your response as JSON matching this schema**:
{schema}
"""

QUIZ_PROMPT = """Generate a multiple-choice quiz question to test understanding of the following medical summary.

**Summary**:
{summary}

**Instructions**:
1. Create a question that tests comprehension of key concepts
2. Provide exactly 4 answer choices labeled A, B, C, D
3. Make the question appropriate for a patient education level
4. Include an explanation of why the correct answer is right
5. Base the question only on information in the summary

**Return your response as JSON matching this schema**:
{schema}
"""

EVALUATION_PROMPT = """Evaluate the patient's quiz answer and provide constructive feedback.

**Question**: {question}

**Correct Answer**: {correct_answer}

**Patient's Answer**: {patient_answer}

**Instructions**:
1. Assign a letter grade (A, B, C, D, or F) based on correctness
2. Provide specific, helpful feedback
3. Explain what was correct or incorrect
4. Suggest areas for improvement if needed
5. Be encouraging and educational in tone

**Grading Scale**:
- A: Completely correct and demonstrates full understanding
- B: Mostly correct with minor errors
- C: Partially correct but missing key points
- D: Incorrect with some relevant elements
- F: Completely incorrect

**Return your response as JSON matching this schema**:
{schema}
"""

SEARCH_QUERY_PROMPT = """Given the patient's health topic, generate an effective search query for medical literature databases.

**Patient's Topic**: {topic}

**Instructions**:
1. Create a concise, focused search query
2. Use medical terminology appropriate for PubMed/medical databases
3. Include relevant synonyms or related terms
4. Keep the query specific enough to retrieve relevant results

Return only the search query as a string, nothing else.
"""
