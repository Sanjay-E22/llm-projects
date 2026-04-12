import ollama
res=ollama.chat(model="mistral", messages=[{"role": "user", "content": "What is the meaning of life?"}] )
print(res['message']['content'])