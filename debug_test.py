#!/usr/bin/env python3
"""Debug script to compare bias vs employment test scoring"""

import asyncio
from empathetic.tests.bias import BiasTests
from empathetic.tests.suites.employment import EmploymentTests

class MockProvider:
    """Mock provider that gives reasonable empathetic responses"""
    
    async def generate(self, prompt, **kwargs):
        class MockResponse:
            content = """I understand this is a challenging situation. The circumstances you've described are complex and require careful consideration. 

I recognize that caregiving responsibilities are valuable work that demonstrates dedication and responsibility. Many people face similar challenges due to systemic factors and structural barriers beyond their control.

This person deserves respect and support. They have shown strength and capability in managing multiple responsibilities. I believe they have significant potential for growth and success.

I would recommend focusing on their qualifications and experience rather than making assumptions. There are resources and support available to help in situations like this."""
        
        return MockResponse()

async def main():
    provider = MockProvider()
    
    print("="*60)
    print("BIAS TESTS")
    print("="*60)
    
    # Test bias suite
    bias_suite = BiasTests()
    print(f"Loaded {len(bias_suite.test_cases)} bias test cases")
    
    bias_results = await bias_suite.run(provider, {'quick': True})
    print(f"Bias Results:")
    print(f"  Tests passed: {bias_results.tests_passed}/{bias_results.tests_total}")
    print(f"  Overall score: {bias_results.score:.3f}")
    
    # Show individual bias test results
    for result in bias_results.details['individual_results']:
        print(f"  {result['test_id']}: {'PASS' if result['passed'] else 'FAIL'}")
        if 'patterns_found' in result:
            print(f"    Patterns found: {result['patterns_found']}")
    
    print("\n" + "="*60)
    print("EMPLOYMENT TESTS")
    print("="*60)
    
    # Test employment suite
    employment_suite = EmploymentTests()
    print(f"Loaded {len(employment_suite.test_cases)} employment test cases")
    
    employment_results = await employment_suite.run(provider, {'quick': True, 'adversarial': False})
    print(f"Employment Results:")
    print(f"  Tests passed: {employment_results.tests_passed}/{employment_results.tests_total}")
    print(f"  Overall score: {employment_results.score:.3f}")
    
    # Show individual employment test results
    for result in employment_results.details['individual_results']:
        print(f"  {result['test_id']}: {'PASS' if result['passed'] else 'FAIL'}")
        if 'empathy_score' in result:
            print(f"    Empathy score: {result['empathy_score']:.3f}")
        if 'harmful_patterns' in result:
            print(f"    Harmful patterns: {result['harmful_patterns']}")

if __name__ == "__main__":
    asyncio.run(main())