import sqlite3
from datetime import datetime

def view_conversations():
    try:
        with sqlite3.connect("c:/Documents/projects/AI/data/conversations.db") as conn:
            cursor = conn.execute("SELECT * FROM conversations ORDER BY timestamp DESC")
            conversations = cursor.fetchall()
            
            print("\nConversation History:")
            print("-" * 80)
            for conv in conversations:
                print(f"Time: {conv[3]}")
                print(f"User: {conv[1]}")
                print(f"AI: {conv[2]}")
                print(f"Source: {conv[4]} (Session: {conv[5]})")
                print("-" * 80)
    except sqlite3.Error as e:
        print(f"Database error: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    view_conversations()