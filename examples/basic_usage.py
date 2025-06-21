#!/usr/bin/env python3
"""
Basic usage example for Empathetic AI Testing Framework
"""

import asyncio
import os

from empathetic.core.tester import Tester
from empathetic.reports.generator import ReportGenerator


async def main():
    """Basic usage example"""

    # Check if API key is set
    if not os.getenv("OPENAI_API_KEY"):
        print("Please set OPENAI_API_KEY environment variable")
        return

    print("🧠 Empathetic AI Testing Framework")
    print("=" * 40)

    # Initialize tester
    tester = Tester()

    # Test a model
    model_name = "gpt-3.5-turbo"
    print(f"Testing model: {model_name}")

    try:
        # Run bias tests
        results = await tester.run_tests(
            model=model_name,
            suites=["bias"],
            verbose=True
        )

        print("\n📊 Results Summary:")
        print(f"Model: {results.model}")
        print(f"Overall Score: {results.overall_score:.3f}")
        print(f"Timestamp: {results.timestamp}")

        # Show suite results
        for suite_name, result in results.suite_results.items():
            print(f"\n{suite_name.title()} Suite:")
            print(f"  Score: {result.score:.3f}")
            print(f"  Tests: {result.tests_passed}/{result.tests_total}")

            if result.recommendations:
                print("  Recommendations:")
                for rec in result.recommendations:
                    print(f"    • {rec}")

        # Generate report
        print("\n📄 Generating report...")
        generator = ReportGenerator()
        report_path = generator.create(results, format='html')
        print(f"Report saved: {report_path}")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
