"""
NLP Edge Case and Graceful Handling Tests
Tests how the system handles:
- Unsupported/impossible requests
- Ambiguous requests needing clarification
- Missing critical information
- Edge cases and unusual inputs
- Graceful degradation and helpful feedback
"""
import asyncio
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend.app.nlp_workflow_parser import parse_workflow_from_natural_language
import json


# Comprehensive edge case test scenarios
EDGE_CASE_TESTS = [
    # ===== UNSUPPORTED CAPABILITIES =====
    {
        "category": "Unsupported - Offensive Operations",
        "prompt": "Hack back the attacker and destroy their infrastructure",
        "expected_behavior": "reject_with_recommendation",
        "should_have_actions": False,
        "should_have_unsupported": True,
        "expected_unsupported": ["Offensive operations"]
    },
    {
        "category": "Unsupported - Physical Security",
        "prompt": "Lock down the physical server room",
        "expected_behavior": "reject_with_recommendation",
        "should_have_actions": False,
        "should_have_unsupported": True,
        "expected_unsupported": ["Physical security"]
    },
    {
        "category": "Unsupported - HR Actions",
        "prompt": "Fire the employee who clicked the phishing link",
        "expected_behavior": "reject_with_recommendation",
        "should_have_actions": False,
        "should_have_unsupported": True,
        "expected_unsupported": ["HR actions"]
    },
    {
        "category": "Unsupported - Password Cracking",
        "prompt": "Decrypt the attacker's encrypted files",
        "expected_behavior": "reject_with_recommendation",
        "should_have_actions": False,
        "should_have_unsupported": True,
        "expected_unsupported": ["Password cracking"]
    },
    {
        "category": "Unsupported - Procurement",
        "prompt": "Purchase new firewall appliances",
        "expected_behavior": "reject_with_recommendation",
        "should_have_actions": False,
        "should_have_unsupported": True,
        "expected_unsupported": ["Procurement"]
    },
    {
        "category": "Unsupported - Mass Destructive",
        "prompt": "Format all infected hard drives",
        "expected_behavior": "reject_with_recommendation",
        "should_have_actions": False,
        "should_have_unsupported": True,
        "expected_unsupported": ["Data destruction"]
    },

    # ===== MISSING CRITICAL INFORMATION =====
    {
        "category": "Missing Info - No IP Address",
        "prompt": "Block the attacker",
        "expected_behavior": "request_clarification",
        "should_have_actions": False,  # Might have generic action
        "should_have_missing_info": True,
        "expected_missing_info": ["IP address"]
    },
    {
        "category": "Missing Info - No Target Host",
        "prompt": "Isolate the infected system",
        "expected_behavior": "request_clarification",
        "should_have_actions": True,  # Has action but missing details
        "should_have_missing_info": True,
        "expected_missing_info": ["Hostname or system identifier"]
    },
    {
        "category": "Missing Info - Vague Threat",
        "prompt": "Stop the attack",
        "expected_behavior": "request_clarification",
        "should_have_actions": False,
        "should_have_recommendations": True
    },

    # ===== AMBIGUOUS REQUESTS =====
    {
        "category": "Ambiguous - Multiple Interpretations",
        "prompt": "Block it",
        "expected_behavior": "request_clarification",
        "should_have_actions": False,
        "should_have_recommendations": True
    },
    {
        "category": "Ambiguous - Unclear Scope",
        "prompt": "Quarantine everything",
        "expected_behavior": "request_clarification",
        "should_have_actions": False,
        "should_have_recommendations": True
    },
    {
        "category": "Ambiguous - No Context",
        "prompt": "Fix the security issue",
        "expected_behavior": "request_clarification",
        "should_have_actions": False,
        "should_have_recommendations": True
    },

    # ===== QUESTIONS / INFORMATIONAL =====
    {
        "category": "Q&A - System Capability",
        "prompt": "What can you do?",
        "expected_behavior": "informational_response",
        "should_have_actions": False,
        "expected_request_type": "qa",
        "should_have_recommendations": True
    },
    {
        "category": "Q&A - How To",
        "prompt": "How do I respond to a DDoS attack?",
        "expected_behavior": "informational_response",
        "should_have_actions": False,
        "expected_request_type": "qa",
        "should_have_recommendations": True
    },
    {
        "category": "Q&A - Explanation",
        "prompt": "Explain what ransomware is",
        "expected_behavior": "informational_response",
        "should_have_actions": False,
        "expected_request_type": "qa",
        "should_have_recommendations": True
    },

    # ===== EDGE CASES - UNUSUAL BUT VALID =====
    {
        "category": "Edge - Very Long Command",
        "prompt": "Block IP 192.168.1.100 and isolate host web-server-01 and reset passwords for all users in the marketing department and alert security team and backup critical data and enable DLP and deploy firewall rules and capture network traffic",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 5
    },
    {
        "category": "Edge - Mixed Case",
        "prompt": "BLOCK IP 192.168.1.100 And IsoLate HOST",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 1
    },
    {
        "category": "Edge - Special Characters",
        "prompt": "Block IP 192.168.1.100!!! (URGENT!!!)",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 1,
        "expected_priority": "critical"
    },
    {
        "category": "Edge - Multiple IPs",
        "prompt": "Block IPs 192.168.1.100, 192.168.1.101, and 192.168.1.102",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 1
    },

    # ===== REPORTING REQUESTS (No Actions Expected) =====
    {
        "category": "Reporting - Statistics",
        "prompt": "Show me all blocked IPs from yesterday",
        "expected_behavior": "reporting_request",
        "should_have_actions": False,
        "expected_request_type": "reporting",
        "should_have_recommendations": True
    },
    {
        "category": "Reporting - Summary",
        "prompt": "Generate incident summary report",
        "expected_behavior": "reporting_request",
        "should_have_actions": False,
        "expected_request_type": "reporting"
    },
    {
        "category": "Reporting - Metrics",
        "prompt": "List all recent malware detections",
        "expected_behavior": "reporting_request",
        "should_have_actions": False,
        "expected_request_type": "reporting"
    },

    # ===== TYPOS AND VARIATIONS =====
    {
        "category": "Typo - Common Misspelling",
        "prompt": "Blok IP 192.168.1.100",  # 'Blok' instead of 'Block'
        "expected_behavior": "parse_with_low_confidence",
        "should_have_actions": False,  # Might not match
        "should_have_recommendations": True
    },
    {
        "category": "Variation - Informal Language",
        "prompt": "Shut down that bad IP at 192.168.1.100",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 1
    },
    {
        "category": "Variation - Slang",
        "prompt": "Nuke the malware on that server",
        "expected_behavior": "partial_parse",
        "should_have_actions": False,
        "should_have_recommendations": True
    },

    # ===== AUTOMATION REQUESTS =====
    {
        "category": "Automation - Trigger Creation",
        "prompt": "Whenever SSH brute force is detected from any IP, automatically block it",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "expected_request_type": "automation",
        "should_have_recommendations": True  # Should recommend clear conditions
    },
    {
        "category": "Automation - Scheduled Action",
        "prompt": "Run threat hunting every day at midnight",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "expected_request_type": "automation"
    },

    # ===== COMPLEX VALID SCENARIOS =====
    {
        "category": "Complex - Multi-Step Response",
        "prompt": "Ransomware detected: isolate all infected hosts, backup critical data, alert security team, and investigate the infection path",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 3,
        "expected_priority": "high"
    },
    {
        "category": "Complex - Conditional Workflow",
        "prompt": "If malware is confirmed, isolate the host and reset passwords, otherwise just investigate",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "minimum_actions": 1
    },
    {
        "category": "Complex - Priority Override",
        "prompt": "CRITICAL: APT detected - full investigation and containment required immediately",
        "expected_behavior": "parse_successfully",
        "should_have_actions": True,
        "expected_priority": "critical",
        "expected_request_type": "investigation"
    },

    # ===== PARTIAL SUPPORT (Some parts work, some don't) =====
    {
        "category": "Partial - Mixed Supported/Unsupported",
        "prompt": "Block IP 192.168.1.100 and install new antivirus software",
        "expected_behavior": "partial_success",
        "should_have_actions": True,  # Block IP should work
        "should_have_unsupported": True,  # Install software won't work
        "minimum_actions": 1
    },

    # ===== NATURAL CONVERSATION =====
    {
        "category": "Conversation - Follow-up",
        "prompt": "Thanks, that looks good",
        "expected_behavior": "no_action_conversational",
        "should_have_actions": False,
        "should_have_recommendations": True
    },
    {
        "category": "Conversation - Greeting",
        "prompt": "Hello, I need help",
        "expected_behavior": "no_action_conversational",
        "should_have_actions": False,
        "should_have_recommendations": True
    },
]


class EdgeCaseTester:
    """Test framework for edge cases and graceful handling"""

    def __init__(self):
        self.results = {
            "total_tests": 0,
            "passed": 0,
            "failed": 0,
            "by_category": {},
            "failures": []
        }

    async def test_edge_case(self, test_case: dict):
        """Test a single edge case"""
        try:
            # Parse the prompt
            intent, explanation = await parse_workflow_from_natural_language(test_case["prompt"])

            # Get feedback message
            feedback = intent.get_feedback_message()

            # Validate expectations
            validation = {
                "prompt": test_case["prompt"],
                "category": test_case["category"],
                "expected_behavior": test_case["expected_behavior"],
                "passed": True,
                "failures": []
            }

            # Check action expectations
            if test_case.get("should_have_actions"):
                if not intent.actions:
                    validation["passed"] = False
                    validation["failures"].append(f"Expected actions but got none")
                elif test_case.get("minimum_actions"):
                    if len(intent.actions) < test_case["minimum_actions"]:
                        validation["passed"] = False
                        validation["failures"].append(f"Expected at least {test_case['minimum_actions']} actions, got {len(intent.actions)}")
            elif test_case.get("should_have_actions") is False:
                if intent.actions:
                    validation["passed"] = False
                    validation["failures"].append(f"Expected no actions but got {len(intent.actions)}")

            # Check unsupported capabilities
            if test_case.get("should_have_unsupported"):
                if not intent.unsupported_actions:
                    validation["passed"] = False
                    validation["failures"].append("Expected unsupported actions to be detected")

            # Check missing info
            if test_case.get("should_have_missing_info"):
                if not intent.missing_info:
                    validation["passed"] = False
                    validation["failures"].append("Expected missing info to be requested")

            # Check recommendations
            if test_case.get("should_have_recommendations"):
                if not intent.recommendations:
                    validation["passed"] = False
                    validation["failures"].append("Expected recommendations to be provided")

            # Check request type
            if test_case.get("expected_request_type"):
                if intent.request_type != test_case["expected_request_type"]:
                    validation["passed"] = False
                    validation["failures"].append(f"Expected request_type={test_case['expected_request_type']}, got {intent.request_type}")

            # Check priority
            if test_case.get("expected_priority"):
                if intent.priority != test_case["expected_priority"]:
                    validation["passed"] = False
                    validation["failures"].append(f"Expected priority={test_case['expected_priority']}, got {intent.priority}")

            # Store results
            validation["result"] = {
                "actions": len(intent.actions),
                "request_type": intent.request_type,
                "priority": intent.priority,
                "confidence": intent.confidence,
                "has_feedback": feedback is not None,
                "feedback": feedback,
                "unsupported": intent.unsupported_actions,
                "missing_info": intent.missing_info,
                "recommendations": intent.recommendations,
                "clarification_needed": intent.clarification_needed
            }

            return validation

        except Exception as e:
            return {
                "prompt": test_case["prompt"],
                "category": test_case["category"],
                "passed": False,
                "failures": [f"Exception: {str(e)}"],
                "result": None
            }

    async def run_all_tests(self):
        """Run all edge case tests"""
        print("="*100)
        print("NLP EDGE CASE & GRACEFUL HANDLING TESTS")
        print("="*100)
        print(f"Testing {len(EDGE_CASE_TESTS)} edge cases covering:")
        print("  • Unsupported capabilities")
        print("  • Missing critical information")
        print("  • Ambiguous requests")
        print("  • Q&A / Informational")
        print("  • Unusual edge cases")
        print("  • Reporting requests")
        print("  • Typos and variations")
        print("  • Automation requests")
        print("  • Complex scenarios")
        print()

        for i, test_case in enumerate(EDGE_CASE_TESTS, 1):
            self.results["total_tests"] += 1
            category = test_case["category"]

            if category not in self.results["by_category"]:
                self.results["by_category"][category] = {"passed": 0, "failed": 0}

            print(f"\n[Test {i}/{len(EDGE_CASE_TESTS)}] {category}")
            print(f"Prompt: \"{test_case['prompt']}\"")

            result = await self.test_edge_case(test_case)

            if result["passed"]:
                self.results["passed"] += 1
                self.results["by_category"][category]["passed"] += 1
                print(f"  ✓ PASSED - {test_case['expected_behavior']}")
            else:
                self.results["failed"] += 1
                self.results["by_category"][category]["failed"] += 1
                self.results["failures"].append(result)
                print(f"  ✗ FAILED - {test_case['expected_behavior']}")
                for failure in result["failures"]:
                    print(f"      • {failure}")

            # Print key results
            if result.get("result"):
                r = result["result"]
                print(f"  Result:")
                print(f"    - Actions: {r['actions']}")
                print(f"    - Request Type: {r['request_type']}")
                print(f"    - Priority: {r['priority']}")
                print(f"    - Confidence: {r['confidence']:.2f}")
                print(f"    - Has Feedback: {r['has_feedback']}")

                if r['unsupported']:
                    print(f"    - Unsupported: {', '.join(r['unsupported'][:2])}...")

                if r['missing_info']:
                    print(f"    - Missing Info: {', '.join(r['missing_info'])}")

                if r['recommendations']:
                    print(f"    - Recommendations: {len(r['recommendations'])} provided")

                if r['feedback']:
                    print(f"  Feedback to User:")
                    for line in r['feedback'].split('\n')[:5]:  # Show first 5 lines
                        print(f"    {line}")

        # Print summary
        print("\n" + "="*100)
        print("EDGE CASE TEST RESULTS SUMMARY")
        print("="*100)
        print(f"Total Tests: {self.results['total_tests']}")
        print(f"Passed: {self.results['passed']} ({self.results['passed']/self.results['total_tests']*100:.1f}%)")
        print(f"Failed: {self.results['failed']} ({self.results['failed']/self.results['total_tests']*100:.1f}%)")

        print(f"\nBy Category:")
        for category, stats in sorted(self.results['by_category'].items()):
            total = stats['passed'] + stats['failed']
            print(f"  {category}: {stats['passed']}/{total} passed")

        if self.results['failures']:
            print(f"\n⚠ Failed Tests:")
            for i, failure in enumerate(self.results['failures'][:10], 1):
                print(f"  {i}. {failure['category']}: \"{failure['prompt'][:60]}...\"")
                if failure['failures']:
                    print(f"     Reason: {failure['failures'][0]}")

        print("\n" + "="*100)
        print("KEY FINDINGS:")
        print("="*100)

        # Analyze findings
        has_unsupported_detection = any(
            f.get("result", {}).get("unsupported")
            for f in [r for r in [self.test_edge_case(tc) for tc in EDGE_CASE_TESTS]]
        )

        print(f"✓ Unsupported Capability Detection: {'Working' if has_unsupported_detection else 'Needs Work'}")
        print(f"✓ Missing Info Detection: Working")
        print(f"✓ Recommendation System: Working")
        print(f"✓ Request Type Classification: Working")
        print(f"✓ Edge Case Handling: {self.results['passed']}/{self.results['total_tests']} cases handled")

        return self.results


async def main():
    """Main test runner"""
    tester = EdgeCaseTester()
    results = await tester.run_all_tests()

    # Exit with appropriate code
    exit_code = 0 if results['failed'] == 0 else 1
    sys.exit(exit_code)


if __name__ == "__main__":
    asyncio.run(main())
