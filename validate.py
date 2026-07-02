"""
Phase 14 – Validation
=====================
Run edge-case and logic checks on the prediction pipeline.

Usage:
    python -m src.validate
"""
from dataclasses import dataclass

from src.predict import format_inr, predict_price, validate_inputs


@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str


def _base_inputs(**overrides):
    defaults = {
        "area": 1800,
        "bedrooms": 3,
        "bathrooms": 2,
        "garage": 1,
        "age": 10,
        "location": "Urban",
    }
    defaults.update(overrides)
    return defaults


def _report_price(amount: float) -> str:
    """Console-safe price string for validation reports."""
    return format_inr(amount).replace("\u20b9", "INR ")


def test_small_area() -> TestResult:
    """A small but valid area should produce a lower price than a large house."""
    small = predict_price(**_base_inputs(area=600))
    large = predict_price(**_base_inputs(area=4500))
    passed = 0 < small < large
    return TestResult(
        "Small Area",
        passed,
        f"600 sq ft -> {_report_price(small)} | 4500 sq ft -> {_report_price(large)} "
        f"({'small < large' if passed else 'NOT logical'})",
    )


def test_large_area() -> TestResult:
    """A large valid area should predict a high, positive price."""
    price = predict_price(**_base_inputs(area=4500))
    passed = price > 15_000_000
    return TestResult(
        "Large Area",
        passed,
        f"4500 sq ft -> {_report_price(price)} ({'positive & high' if passed else 'unexpected'})",
    )


def test_negative_input() -> TestResult:
    """Negative area must be rejected before prediction."""
    try:
        predict_price(**_base_inputs(area=-500))
        return TestResult("Negative Input", False, "Negative area was accepted (should be blocked).")
    except ValueError as exc:
        return TestResult("Negative Input", True, f"Correctly rejected: {exc}")


def test_zero_bedrooms() -> TestResult:
    """Zero bedrooms is invalid for this model."""
    try:
        predict_price(**_base_inputs(bedrooms=0))
        return TestResult("Zero Bedrooms", False, "0 bedrooms was accepted (should be blocked).")
    except ValueError as exc:
        return TestResult("Zero Bedrooms", True, f"Correctly rejected: {exc}")


def test_maximum_area() -> TestResult:
    """Maximum allowed area should return a valid, positive prediction."""
    price = predict_price(**_base_inputs(area=10000))
    passed = price > 0
    return TestResult(
        "Maximum Area",
        passed,
        f"10000 sq ft -> {_report_price(price)} ({'valid prediction' if passed else 'invalid'})",
    )


def test_wrong_values() -> TestResult:
    """Invalid location strings must be rejected."""
    try:
        validate_inputs(
            area=1800,
            bedrooms=3,
            bathrooms=2,
            garage=1,
            age=10,
            location="Suburban",
        )
        return TestResult("Wrong Values", False, "Invalid location was accepted.")
    except ValueError as exc:
        return TestResult("Wrong Values", True, f"Correctly rejected: {exc}")


def test_logical_trends() -> TestResult:
    """More bedrooms and Urban location should increase price vs baseline."""
    baseline = predict_price(**_base_inputs(area=2500, bedrooms=2, location="Rural"))
    more_bedrooms = predict_price(**_base_inputs(area=2500, bedrooms=5, location="Rural"))
    urban = predict_price(**_base_inputs(area=2500, bedrooms=2, location="Urban"))

    area_up = predict_price(**_base_inputs(area=3500, bedrooms=2, location="Rural")) > baseline
    beds_up = more_bedrooms >= baseline
    urban_up = urban > baseline
    passed = area_up and beds_up and urban_up

    return TestResult(
        "Logical Trends",
        passed,
        f"Rural 2500 sq ft/2 bed -> {_report_price(baseline)} | "
        f"+bedrooms -> {_report_price(more_bedrooms)} | "
        f"Urban -> {_report_price(urban)} | "
        f"area_up={area_up}, beds_up={beds_up}, urban_up={urban_up}",
    )


def run_validation() -> list[TestResult]:
    tests = [
        test_small_area,
        test_large_area,
        test_negative_input,
        test_zero_bedrooms,
        test_maximum_area,
        test_wrong_values,
        test_logical_trends,
    ]
    return [test() for test in tests]


def print_report(results: list[TestResult]) -> None:
    print("\nPhase 14 - Validation Report")
    print("=" * 72)
    passed = sum(r.passed for r in results)
    total = len(results)

    for result in results:
        status = "PASS" if result.passed else "FAIL"
        print(f"[{status}] {result.name}")
        print(f"       {result.detail}")

    print("-" * 72)
    print(f"Result: {passed}/{total} tests passed")
    if passed == total:
        print("All validations passed. Predictions behave logically.")
    else:
        print("Some validations failed. Review the details above.")


if __name__ == "__main__":
    print_report(run_validation())
