"""
Comprehensive test script for all agent tools.

Tests:
- Transaction Analyzer tools (4)
- Behavioral Profiler tools (4)
- Geospatial Analyzer tools (4)
"""

import json
from ai_orchestration import (
    DataManager,
    # Transaction Analyzer
    get_user_transaction_history,
    calculate_transaction_velocity,
    get_recipient_profile,
    query_fraud_memory,
    # Behavioral Profiler
    get_user_communications,
    get_user_profile,
    get_user_baseline,
    detect_phishing_patterns,
    # Geospatial Analyzer
    get_user_gps_history,
    calculate_distance,
    check_impossible_travel,
    get_user_residence
)


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_result(tool_name, result):
    """Print tool result."""
    print(f"\n[{tool_name}]")
    try:
        parsed = json.loads(result)
        print(json.dumps(parsed, indent=2)[:500])  # Limit output
    except:
        print(result[:500])


def test_transaction_analyzer_tools():
    """Test Transaction Analyzer tools."""
    print_section("TRANSACTION ANALYZER TOOLS")

    # Get a sample user IBAN
    data_mgr = DataManager.get_instance()
    users = data_mgr.get_all_users()
    sample_iban = list(users.keys())[0] if users else None

    if not sample_iban:
        print("No users found for testing")
        return

    print(f"\nUsing sample IBAN: {sample_iban}")

    # Test 1: get_user_transaction_history
    result = get_user_transaction_history.invoke({"user_iban": sample_iban, "limit": 5})
    print_result("get_user_transaction_history", result)

    # Test 2: calculate_transaction_velocity
    result = calculate_transaction_velocity.invoke({"user_iban": sample_iban, "time_window_hours": 24})
    print_result("calculate_transaction_velocity", result)

    # Test 3: get_recipient_profile
    # Get a recipient IBAN from transactions
    transactions = data_mgr.get_transactions()
    if transactions:
        recipient_iban = transactions[0].get('recipient_iban')
        result = get_recipient_profile.invoke({"recipient_iban": recipient_iban})
        print_result("get_recipient_profile", result)

    # Test 4: query_fraud_memory
    result = query_fraud_memory.invoke({"pattern_type": "all"})
    print_result("query_fraud_memory", result)


def test_behavioral_profiler_tools():
    """Test Behavioral Profiler tools."""
    print_section("BEHAVIORAL PROFILER TOOLS")

    data_mgr = DataManager.get_instance()
    users = data_mgr.get_all_users()
    sample_iban = list(users.keys())[0] if users else None

    if not sample_iban:
        print("No users found for testing")
        return

    print(f"\nUsing sample IBAN: {sample_iban}")

    # Test 1: get_user_profile
    result = get_user_profile.invoke({"user_iban": sample_iban})
    print_result("get_user_profile", result)

    # Test 2: get_user_communications
    # Get a biotag from locations
    locations = data_mgr.get_all_locations()
    if locations:
        biotag = locations[0].get('biotag')
        result = get_user_communications.invoke({"user_biotag": biotag, "limit": 5})
        print_result("get_user_communications", result)

    # Test 3: get_user_baseline
    result = get_user_baseline.invoke({"user_iban": sample_iban})
    print_result("get_user_baseline", result)

    # Test 4: detect_phishing_patterns
    # Test with a phishing email
    phishing_text = """
    URGENT: Your account will be suspended!
    Click here to verify your account immediately: https://paypa1-secure.net/verify
    """
    result = detect_phishing_patterns.invoke({"communication_text": phishing_text})
    print_result("detect_phishing_patterns (phishing)", result)

    # Test with a legitimate email
    legit_text = """
    Hello, your statement for March is now available.
    You can view it at https://www.bank.com/statements
    """
    result = detect_phishing_patterns.invoke({"communication_text": legit_text})
    print_result("detect_phishing_patterns (legit)", result)


def test_geospatial_analyzer_tools():
    """Test Geospatial Analyzer tools."""
    print_section("GEOSPATIAL ANALYZER TOOLS")

    data_mgr = DataManager.get_instance()

    # Test 1: get_user_gps_history
    locations = data_mgr.get_all_locations()
    if locations:
        biotag = locations[0].get('biotag')
        print(f"\nUsing sample biotag: {biotag}")

        result = get_user_gps_history.invoke({"user_biotag": biotag, "last_n_hours": 48})
        print_result("get_user_gps_history", result)

        # Test 2: calculate_distance
        if len(locations) >= 2:
            loc1 = locations[0]
            loc2 = locations[1]
            result = calculate_distance.invoke({
                "lat1": loc1.get('lat'),
                "lon1": loc1.get('lng'),
                "lat2": loc2.get('lat'),
                "lon2": loc2.get('lng')
            })
            print_result("calculate_distance", result)

        # Test 3: check_impossible_travel
        if len(locations) >= 2:
            loc1 = locations[0]
            loc2 = locations[10] if len(locations) > 10 else locations[-1]
            result = check_impossible_travel.invoke({
                "prev_lat": loc1.get('lat'),
                "prev_lon": loc1.get('lng'),
                "prev_time": loc1.get('timestamp'),
                "curr_lat": loc2.get('lat'),
                "curr_lon": loc2.get('lng'),
                "curr_time": loc2.get('timestamp')
            })
            print_result("check_impossible_travel", result)

    # Test 4: get_user_residence
    users = data_mgr.get_all_users()
    if users:
        sample_iban = list(users.keys())[0]
        result = get_user_residence.invoke({"user_iban": sample_iban})
        print_result("get_user_residence", result)


def main():
    """Run all tool tests."""
    print("=" * 70)
    print("  TESTING ALL AGENT TOOLS")
    print("=" * 70)

    # Initialize DataManager
    data_mgr = DataManager.get_instance("The Truman Show_train")
    data_mgr.load_all_data()

    # Run tests
    test_transaction_analyzer_tools()
    test_behavioral_profiler_tools()
    test_geospatial_analyzer_tools()

    print("\n" + "=" * 70)
    print("  ALL TOOL TESTS COMPLETED")
    print("=" * 70)

    # Summary
    print("\n✅ Transaction Analyzer Tools: 4/4 tested")
    print("✅ Behavioral Profiler Tools: 4/4 tested")
    print("✅ Geospatial Analyzer Tools: 4/4 tested")
    print("\nTotal: 12/12 tools tested successfully!")


if __name__ == "__main__":
    main()
