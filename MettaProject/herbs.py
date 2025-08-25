from hyperon import MeTTa

# Initialize MeTTa engine
metta = MeTTa()

# Pre-load some example knowledge (Herbs domain)
metta.run('''
("Herb" "Basil")
(benefit "Basil" "Anti-inflammatory")
(benefit "Basil" "Boosts digestion")
(used-in "Basil" "Italian cuisine")

("Herb" "Rosemary")
(benefit "Rosemary" "Improves memory")
(benefit "Rosemary" "Supports circulation")
(used-in "Rosemary" "Roasted meats,cooking rice,cooking tea")
''')

# Function to query knowledge
def query_knowledge(entity, relation):
    results = metta.run(f'({relation} "{entity}" ?x)')
    return [str(r) for r in results] if results else []

# Function to add new knowledge
def add_knowledge(fact):
    """
    fact should be in the form: (relation "Entity" "Value")
    Example: (benefit "Ginger" "Boosts immunity")
    """
    try:
        metta.run(fact)
        return f"✅ Added: {fact}"
    except Exception as e:
        return f"❌ Error adding fact: {e}"

# Chatbot loop
def chatbot():
    print("🌱 HerbWise Chatbot (MeTTa + Python)")
    print("Type 'exit' to quit.")
    print("You can ask things like: benefits of Basil, uses of Rosemary")
    print("Or add knowledge: add (benefit \"Ginger\" \"Boosts immunity\")")
    print("-----------------------------------------------------------")

    while True:
        user_input = input("You: ")

        if user_input.lower() == "exit":
            print("👋 Goodbye!")
            break

        # Handle adding knowledge
        if user_input.startswith("add "):
            fact = user_input.replace("add:", "").strip()
            metta.run(fact)
            print("Bot: Got it, I’ve added that to my knowledge base!")
            continue

        # Simple query parser
        if "benefit" in user_input.lower():
            herb = user_input.split()[-1].capitalize()
            facts = query_knowledge(herb, "benefit")
            if facts:
                print(f"Bot: {herb} is known for: {', '.join(facts)}.")
            else:
                print(f"Bot: Sorry, I don’t know how {herb} is used yet.")

        elif user_input.lower():
            herb = user_input.split()[-1].capitalize()
            facts = query_knowledge(herb, "used-in")
            if facts:
                print(f"Bot: {herb} is commonly used in: {', '.join(facts)}.")
            else:
                print(f"Bot: Sorry, I don’t know how {herb} is used yet.")

        else:
            print("Bot: Try 'add:' or 'ask:' to interact with my knowledge base.")

# Run chatbot
if __name__ == "__main__":
    chatbot()

