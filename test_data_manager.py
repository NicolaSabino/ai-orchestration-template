"""
Quick test script for DataManager class.
"""

from ai_orchestration import DataManager

def test_data_manager():
    """Test DataManager loading and access methods."""

    print("=" * 70)
    print("Testing DataManager")
    print("=" * 70)

    # Initialize DataManager with data directory
    data_mgr = DataManager.get_instance("The Truman Show_train")

    # Load all data
    data_mgr.load_all_data()

    print("\n" + "=" * 70)
    print("Testing Data Access Methods")
    print("=" * 70)

    # Test transactions
    transactions = data_mgr.get_transactions()
    print(f"\n1. Transactions: {len(transactions)} loaded")
    if transactions:
        print(f"   First transaction ID: {transactions[0].get('transaction_id')}")
        print(f"   Sender IBAN: {transactions[0].get('sender_iban')}")
        print(f"   Amount: {transactions[0].get('amount')}")

    # Test users
    users = data_mgr.get_all_users()
    print(f"\n2. Users: {len(users)} loaded")
    if users:
        first_iban = list(users.keys())[0]
        first_user = users[first_iban]
        print(f"   First user IBAN: {first_iban}")
        print(f"   Name: {first_user.get('first_name')} {first_user.get('last_name')}")
        print(f"   Job: {first_user.get('job')}")

        # Test get_user method
        user = data_mgr.get_user(first_iban)
        print(f"   get_user() works: {user is not None}")

    # Test locations
    all_locations = data_mgr.get_all_locations()
    print(f"\n3. GPS Locations: {len(all_locations)} loaded")
    if all_locations:
        first_loc = all_locations[0]
        biotag = first_loc.get('biotag')
        print(f"   First location biotag: {biotag}")
        print(f"   Coordinates: ({first_loc.get('lat')}, {first_loc.get('lng')})")

        # Test get_user_gps method
        user_gps = data_mgr.get_user_gps(biotag)
        print(f"   GPS points for {biotag}: {len(user_gps)}")

    # Test communications
    all_mails = data_mgr.get_all_mails()
    all_sms = data_mgr.get_all_sms()
    print(f"\n4. Communications:")
    print(f"   Mails: {len(all_mails)} loaded")
    print(f"   SMS: {len(all_sms)} loaded")

    # Test singleton pattern
    print("\n" + "=" * 70)
    print("Testing Singleton Pattern")
    print("=" * 70)

    data_mgr2 = DataManager.get_instance("The Truman Show_train")
    print(f"Singleton works: {data_mgr is data_mgr2}")

    print("\n" + "=" * 70)
    print("All tests completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    test_data_manager()
