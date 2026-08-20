"""
Medical Calculator Tool for HealthBot Phase 4.

Implements medical calculations:
- BMI (Body Mass Index)
- Medication dosage
- Creatinine clearance (Cockcroft-Gault formula)

These calculations demonstrate non-RAG capabilities of the GenAI orchestration system.
"""

from typing import Dict, Any, Literal


def calculate_bmi(weight_kg: float, height_m: float) -> Dict[str, Any]:
    """
    Calculate Body Mass Index (BMI).

    Args:
        weight_kg: Weight in kilograms
        height_m: Height in meters

    Returns:
        dict with bmi, category, and interpretation

    Example:
        >>> calculate_bmi(70, 1.75)
        {'bmi': 22.9, 'category': 'normal', 'interpretation': '...'}
    """
    if weight_kg <= 0 or height_m <= 0:
        return {
            "success": False,
            "error": "Weight and height must be positive numbers",
        }

    if weight_kg > 500 or height_m > 3.0:
        return {
            "success": False,
            "error": "Invalid values: weight must be ≤500kg, height must be ≤3.0m",
        }

    bmi = weight_kg / (height_m ** 2)
    category = categorize_bmi(bmi)

    return {
        "success": True,
        "bmi": round(bmi, 1),
        "category": category,
        "interpretation": f"BMI of {bmi:.1f} indicates {category} weight status",
        "calculation_method": "BMI = weight(kg) / height(m)²",
    }


def categorize_bmi(bmi: float) -> str:
    """
    Categorize BMI according to WHO standards.

    WHO BMI Categories:
    - Underweight: <18.5
    - Normal: 18.5-24.9
    - Overweight: 25.0-29.9
    - Obese: ≥30.0
    """
    if bmi < 18.5:
        return "underweight"
    elif bmi < 25.0:
        return "normal"
    elif bmi < 30.0:
        return "overweight"
    else:
        return "obese"


def calculate_dosage(weight_kg: float, dose_per_kg: float) -> Dict[str, Any]:
    """
    Calculate medication dosage based on weight.

    Args:
        weight_kg: Patient weight in kilograms
        dose_per_kg: Medication dose per kilogram (mg/kg)

    Returns:
        dict with total dose and interpretation

    Example:
        >>> calculate_dosage(70, 5)
        {'total_dose_mg': 350.0, 'interpretation': '...'}
    """
    if weight_kg <= 0 or dose_per_kg <= 0:
        return {
            "success": False,
            "error": "Weight and dose must be positive numbers",
        }

    if weight_kg > 500:
        return {
            "success": False,
            "error": "Invalid weight: must be ≤500kg",
        }

    if dose_per_kg > 100:
        return {
            "success": False,
            "error": "Dose seems unusually high. Please verify the prescription.",
        }

    total_dose = weight_kg * dose_per_kg

    return {
        "success": True,
        "total_dose_mg": round(total_dose, 1),
        "interpretation": f"For a {weight_kg}kg patient at {dose_per_kg}mg/kg: {total_dose:.1f}mg total dose",
        "calculation_method": "Total dose = weight(kg) × dose_per_kg(mg/kg)",
        "disclaimer": "This is a calculation only. Always verify with healthcare provider.",
    }


def calculate_creatinine_clearance(
    age: int,
    weight_kg: float,
    serum_creatinine_mg_dl: float,
    sex: Literal["male", "female"]
) -> Dict[str, Any]:
    """
    Calculate creatinine clearance using Cockcroft-Gault formula.

    Used to estimate kidney function.

    Formula:
    CrCl = ((140 - age) × weight(kg)) / (72 × serum_cr) × (0.85 if female)

    Args:
        age: Patient age in years
        weight_kg: Weight in kilograms
        serum_creatinine_mg_dl: Serum creatinine in mg/dL
        sex: "male" or "female"

    Returns:
        dict with creatinine clearance and interpretation

    Example:
        >>> calculate_creatinine_clearance(65, 70, 1.2, "male")
        {'crcl_ml_min': 61.3, 'interpretation': '...'}
    """
    if age <= 0 or weight_kg <= 0 or serum_creatinine_mg_dl <= 0:
        return {
            "success": False,
            "error": "Age, weight, and serum creatinine must be positive numbers",
        }

    if age > 120:
        return {
            "success": False,
            "error": "Invalid age: must be ≤120 years",
        }

    if weight_kg > 500:
        return {
            "success": False,
            "error": "Invalid weight: must be ≤500kg",
        }

    if sex not in ["male", "female"]:
        return {
            "success": False,
            "error": "Sex must be 'male' or 'female'",
        }

    # Cockcroft-Gault formula
    crcl = ((140 - age) * weight_kg) / (72 * serum_creatinine_mg_dl)

    # Apply 0.85 correction factor for females
    if sex == "female":
        crcl *= 0.85

    # Interpret kidney function
    interpretation = interpret_crcl(crcl)

    return {
        "success": True,
        "crcl_ml_min": round(crcl, 1),
        "interpretation": interpretation,
        "calculation_method": "Cockcroft-Gault formula",
        "formula": "CrCl = ((140 - age) × weight) / (72 × serum_cr) × (0.85 if female)",
        "disclaimer": "This is an estimate. Actual kidney function should be assessed by healthcare provider.",
    }


def interpret_crcl(crcl: float) -> str:
    """
    Interpret creatinine clearance value.

    Stages of kidney function:
    - Normal: ≥90 mL/min
    - Mild decrease: 60-89 mL/min
    - Moderate decrease: 30-59 mL/min
    - Severe decrease: 15-29 mL/min
    - Kidney failure: <15 mL/min
    """
    if crcl >= 90:
        return f"CrCl of {crcl:.1f} mL/min indicates normal kidney function"
    elif crcl >= 60:
        return f"CrCl of {crcl:.1f} mL/min indicates mildly decreased kidney function"
    elif crcl >= 30:
        return f"CrCl of {crcl:.1f} mL/min indicates moderately decreased kidney function"
    elif crcl >= 15:
        return f"CrCl of {crcl:.1f} mL/min indicates severely decreased kidney function"
    else:
        return f"CrCl of {crcl:.1f} mL/min indicates kidney failure - immediate medical attention required"


def medical_calculator_tool(
    calculation_type: Literal["bmi", "dosage", "creatinine_clearance"],
    **params
) -> Dict[str, Any]:
    """
    Main entry point for medical calculator tool.

    This function routes to the appropriate calculation based on type.

    Args:
        calculation_type: Type of calculation to perform
        **params: Parameters specific to the calculation type

    Returns:
        Dictionary with calculation results

    Examples:
        >>> medical_calculator_tool("bmi", weight_kg=70, height_m=1.75)
        >>> medical_calculator_tool("dosage", weight_kg=70, dose_per_kg=5)
        >>> medical_calculator_tool("creatinine_clearance", age=65, weight_kg=70,
        ...                          serum_creatinine_mg_dl=1.2, sex="male")
    """
    if calculation_type == "bmi":
        required = ["weight_kg", "height_m"]
        if not all(param in params for param in required):
            return {
                "success": False,
                "error": f"BMI calculation requires: {', '.join(required)}",
            }
        return calculate_bmi(params["weight_kg"], params["height_m"])

    elif calculation_type == "dosage":
        required = ["weight_kg", "dose_per_kg"]
        if not all(param in params for param in required):
            return {
                "success": False,
                "error": f"Dosage calculation requires: {', '.join(required)}",
            }
        return calculate_dosage(params["weight_kg"], params["dose_per_kg"])

    elif calculation_type == "creatinine_clearance":
        required = ["age", "weight_kg", "serum_creatinine_mg_dl", "sex"]
        if not all(param in params for param in required):
            return {
                "success": False,
                "error": f"Creatinine clearance calculation requires: {', '.join(required)}",
            }
        return calculate_creatinine_clearance(
            params["age"],
            params["weight_kg"],
            params["serum_creatinine_mg_dl"],
            params["sex"]
        )

    else:
        return {
            "success": False,
            "error": f"Unknown calculation type: {calculation_type}. "
                     "Supported types: bmi, dosage, creatinine_clearance",
        }
