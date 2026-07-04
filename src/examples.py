from backend.utils.prompt import clarify_user_intent

raw_prompt = "Write a text in 4chan greentext style about the manager of a bottomless pit. The absurder and funnier, the better"
output = clarify_user_intent(raw_prompt, "cloud")
print(output.json())
raw_prompt = (
    "Create a tail recursive function to compute the factorial of a number in scala"
)
output = clarify_user_intent(raw_prompt, "cloud")
print(output.json())
