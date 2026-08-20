"""
Tests for medical calculator tool (Phase 4).

Tests BMI, dosage, and creatinine clearance calculations.
"""

import pytest

from healthbot.tools.medical_calculator import (
    calculate_bmi,
    categorize_bmi,
    calculate_dosage,
    calculate_creatinine_clearance,
    medical_calculator_tool,
)


class TestBMICalculator:
    """Test BMI calculation functionality."""

    def test_normal_bmi(self):
        """Test BMI calculation in normal range."""
        result = calculate_bmi(weight_kg=70, height_m=1.75)
        assert result["success"]
        assert result["bmi"] == 22.9  # 70 / (1.75^2) = 22.86
        assert result["category"] == "normal"
        assert "normal" in result["interpretation"].lower()

    def test_underweight_bmi(self):
        """Test BMI calculation for underweight category."""
        result = calculate_bmi(weight_kg=50, height_m=1.75)
        assert result["success"]
        assert result["bmi"] == 16.3
        assert result["category"] == "underweight"

    def test_overweight_bmi(self):
        """Test BMI calculation for overweight category."""
        result = calculate_bmi(weight_kg=85, height_m=1.75)
        assert result["success"]
        assert result["bmi"] == 27.8
        assert result["category"] == "overweight"

    def test_obese_bmi(self):
        """Test BMI calculation for obese category."""
        result = calculate_bmi(weight_kg=100, height_m=1.75)
        assert result["success"]
        assert result["bmi"] == 32.7
        assert result["category"] == "obese"

    def test_bmi_boundary_normal_overweight(self):
        """Test BMI boundary between normal and overweight (25.0)."""
        # Exactly 25.0
        result = calculate_bmi(weight_kg=76.6, height_m=1.75)
        assert result["success"]
        assert result["category"] == "overweight"  # ≥25.0 is overweight

    def test_bmi_invalid_weight(self):
        """Test BMI with invalid weight."""
        result = calculate_bmi(weight_kg=-10, height_m=1.75)
        assert not result["success"]
        assert "error" in result

    def test_bmi_invalid_height(self):
        """Test BMI with invalid height."""
        result = calculate_bmi(weight_kg=70, height_m=0)
        assert not result["success"]
        assert "error" in result

    def test_bmi_unrealistic_values(self):
        """Test BMI with unrealistic but positive values."""
        result = calculate_bmi(weight_kg=600, height_m=1.75)
        assert not result["success"]
        assert "Invalid values" in result["error"]


class TestBMICategorization:
    """Test BMI categorization logic."""

    def test_categorize_underweight(self):
        """Test underweight categorization (<18.5)."""
        assert categorize_bmi(18.4) == "underweight"
        assert categorize_bmi(15.0) == "underweight"

    def test_categorize_normal(self):
        """Test normal weight categorization (18.5-24.9)."""
        assert categorize_bmi(18.5) == "normal"
        assert categorize_bmi(22.0) == "normal"
        assert categorize_bmi(24.9) == "normal"

    def test_categorize_overweight(self):
        """Test overweight categorization (25.0-29.9)."""
        assert categorize_bmi(25.0) == "overweight"
        assert categorize_bmi(27.5) == "overweight"
        assert categorize_bmi(29.9) == "overweight"

    def test_categorize_obese(self):
        """Test obese categorization (≥30.0)."""
        assert categorize_bmi(30.0) == "obese"
        assert categorize_bmi(35.0) == "obese"
        assert categorize_bmi(40.0) == "obese"


class TestDosageCalculator:
    """Test medication dosage calculation."""

    def test_standard_dosage(self):
        """Test standard dosage calculation."""
        result = calculate_dosage(weight_kg=70, dose_per_kg=5)
        assert result["success"]
        assert result["total_dose_mg"] == 350.0
        assert "70kg" in result["interpretation"]
        assert "350" in result["interpretation"]

    def test_pediatric_dosage(self):
        """Test dosage for pediatric patient."""
        result = calculate_dosage(weight_kg=20, dose_per_kg=10)
        assert result["success"]
        assert result["total_dose_mg"] == 200.0

    def test_fractional_dosage(self):
        """Test dosage with fractional dose_per_kg."""
        result = calculate_dosage(weight_kg=65, dose_per_kg=2.5)
        assert result["success"]
        assert result["total_dose_mg"] == 162.5

    def test_dosage_invalid_weight(self):
        """Test dosage with invalid weight."""
        result = calculate_dosage(weight_kg=-5, dose_per_kg=5)
        assert not result["success"]
        assert "error" in result

    def test_dosage_invalid_dose(self):
        """Test dosage with invalid dose_per_kg."""
        result = calculate_dosage(weight_kg=70, dose_per_kg=-5)
        assert not result["success"]
        assert "error" in result

    def test_dosage_unusually_high(self):
        """Test dosage validation for unusually high dose."""
        result = calculate_dosage(weight_kg=70, dose_per_kg=150)
        assert not result["success"]
        assert "unusually high" in result["error"].lower()

    def test_dosage_disclaimer_present(self):
        """Test that medical disclaimer is included."""
        result = calculate_dosage(weight_kg=70, dose_per_kg=5)
        assert result["success"]
        assert "disclaimer" in result
        assert "healthcare provider" in result["disclaimer"].lower()


class TestCreatinineClearance:
    """Test creatinine clearance (Cockcroft-Gault) calculation."""

    def test_male_normal_kidney_function(self):
        """Test CrCl calculation for male with normal kidney function."""
        result = calculate_creatinine_clearance(
            age=40, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="male"
        )
        assert result["success"]
        # CrCl = ((140 - 40) * 70) / (72 * 1.0) = 97.2
        assert result["crcl_ml_min"] == 97.2
        assert "normal" in result["interpretation"].lower()

    def test_female_creatinine_clearance(self):
        """Test CrCl calculation for female (0.85 correction factor)."""
        result_male = calculate_creatinine_clearance(
            age=40, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="male"
        )
        result_female = calculate_creatinine_clearance(
            age=40, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="female"
        )

        assert result_female["success"]
        # Female should be 85% of male value
        assert result_female["crcl_ml_min"] == pytest.approx(
            result_male["crcl_ml_min"] * 0.85, rel=0.01
        )

    def test_elderly_decreased_kidney_function(self):
        """Test CrCl for elderly patient with decreased kidney function."""
        result = calculate_creatinine_clearance(
            age=80, weight_kg=65, serum_creatinine_mg_dl=1.8, sex="male"
        )
        assert result["success"]
        # CrCl = ((140 - 80) * 65) / (72 * 1.8) = 30.1
        assert result["crcl_ml_min"] == 30.1
        assert "moderate" in result["interpretation"].lower()

    def test_crcl_severe_kidney_disease(self):
        """Test CrCl interpretation for severe kidney disease."""
        result = calculate_creatinine_clearance(
            age=70, weight_kg=60, serum_creatinine_mg_dl=3.5, sex="female"
        )
        assert result["success"]
        # With these values, CrCl = ((140-70)*60)/(72*3.5)*0.85 = 14.2
        # This is <15, so it's kidney failure, not severe decrease
        assert result["crcl_ml_min"] < 15
        assert "kidney failure" in result["interpretation"].lower()

    def test_crcl_invalid_age(self):
        """Test CrCl with invalid age."""
        result = calculate_creatinine_clearance(
            age=-5, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="male"
        )
        assert not result["success"]
        assert "error" in result

    def test_crcl_invalid_sex(self):
        """Test CrCl with invalid sex value."""
        result = calculate_creatinine_clearance(
            age=40, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="other"
        )
        assert not result["success"]
        assert "must be 'male' or 'female'" in result["error"]

    def test_crcl_unrealistic_age(self):
        """Test CrCl with unrealistic age."""
        result = calculate_creatinine_clearance(
            age=150, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="male"
        )
        assert not result["success"]
        assert "Invalid age" in result["error"]

    def test_crcl_formula_documentation(self):
        """Test that Cockcroft-Gault formula is documented in result."""
        result = calculate_creatinine_clearance(
            age=40, weight_kg=70, serum_creatinine_mg_dl=1.0, sex="male"
        )
        assert result["success"]
        assert "formula" in result
        assert "Cockcroft-Gault" in result["calculation_method"]


class TestMedicalCalculatorTool:
    """Test the main medical calculator tool entry point."""

    def test_bmi_calculation_via_tool(self):
        """Test BMI calculation through main tool interface."""
        result = medical_calculator_tool(
            calculation_type="bmi",
            weight_kg=70,
            height_m=1.75
        )
        assert result["success"]
        assert result["bmi"] == 22.9

    def test_dosage_calculation_via_tool(self):
        """Test dosage calculation through main tool interface."""
        result = medical_calculator_tool(
            calculation_type="dosage",
            weight_kg=70,
            dose_per_kg=5
        )
        assert result["success"]
        assert result["total_dose_mg"] == 350.0

    def test_crcl_calculation_via_tool(self):
        """Test creatinine clearance through main tool interface."""
        result = medical_calculator_tool(
            calculation_type="creatinine_clearance",
            age=40,
            weight_kg=70,
            serum_creatinine_mg_dl=1.0,
            sex="male"
        )
        assert result["success"]
        assert result["crcl_ml_min"] == 97.2

    def test_unknown_calculation_type(self):
        """Test tool with unknown calculation type."""
        result = medical_calculator_tool(
            calculation_type="blood_pressure",
            systolic=120,
            diastolic=80
        )
        assert not result["success"]
        assert "Unknown calculation type" in result["error"]

    def test_missing_required_parameters_bmi(self):
        """Test BMI calculation with missing parameters."""
        result = medical_calculator_tool(
            calculation_type="bmi",
            weight_kg=70
            # Missing height_m
        )
        assert not result["success"]
        assert "requires" in result["error"].lower()

    def test_missing_required_parameters_dosage(self):
        """Test dosage calculation with missing parameters."""
        result = medical_calculator_tool(
            calculation_type="dosage",
            weight_kg=70
            # Missing dose_per_kg
        )
        assert not result["success"]
        assert "requires" in result["error"].lower()

    def test_missing_required_parameters_crcl(self):
        """Test creatinine clearance with missing parameters."""
        result = medical_calculator_tool(
            calculation_type="creatinine_clearance",
            age=40,
            weight_kg=70
            # Missing serum_creatinine_mg_dl and sex
        )
        assert not result["success"]
        assert "requires" in result["error"].lower()


# Mark all tests as Phase 4 calculator tests
pytestmark = pytest.mark.calculator
