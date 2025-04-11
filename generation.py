import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import re



def extract_answer(generated_text):
    """
    Extracts only the final answer from the generated response.
    Stops at the end of the passage (e.g., before function definitions or metadata).
    Handles different formats like:
    - "### Answer is: <actual answer>"
    - "Generated Answer:\n <actual answer>"
    """

    #  1. Check if the text contains "### Answer is:"
    match = re.search(r"### Answer is:\s*(.*)", generated_text, re.DOTALL)
    if match:
        answer = match.group(1).strip()
    else:
        #  2. If not found, check for "Generated Answer:"
        match = re.search(r"Generated Answer:\s*\n*(.*)", generated_text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
        else:
            #  3. If neither pattern is found, return the whole text (fallback)
            return generated_text.strip()

    #  4. Stop extraction when hitting a known separator (e.g., function/class definitions, triple quotes)
    stop_patterns = [r'"""\s*', r'\ndef\s', r'\nclass\s', r'\n\s*\n']  # Stops at """ or def/class or empty line
    for stop_pattern in stop_patterns:
        answer = re.split(stop_pattern, answer, maxsplit=1)[0].strip()

    return answer

def extract_score(generated_text):
    """
    Extracts the numeric grade assigned to the answer.
    Looks for the pattern: "### Answer is graded:\n <number>"
    """
    match = re.search(r"### Answer is graded:\s*(\d+)", generated_text)
    
    if match:
        return int(match.group(1))  # Extract and convert to integer
    else:
        match = re.search(r"### Your Answer should go here:\s*\n*(.*)", generated_text, re.DOTALL)
        if match:
            answer = match.group(1).strip()
            return answer
        else:
            #  3. If neither pattern is found, return the whole text (fallback)
            return None
    
    


def generate_answer(query, retrieved_chunks, MODEL_NAME, tokenizer, model):
    """
    I will give an example case for you:
    example Question:
    Millî Eğitim Bakanlığı Öğretmen Atama ve Yer Değiştirme Yönetmeliğine göre, öğretmenlik görevi dışında bir göreve atanmış 
    olan öğretmenler tekrar öğretmenliğe dönebilir mi?

    Example Answer:
    Evet, bu yönetmeliğin yayımı tarihinden önce Bakanlığın merkez veya taşra teşkilatında eğitim ve öğretim hizmetleri sınıfı 
    dışındaki görevlere isteğiyle atanmış öğretmenler, yönetmeliğin yayımı tarihinden itibaren altı ay içinde başvuruda bulunmaları 
    hâlinde yeniden öğretmenliğe atanabilir.
    """
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    context = "\n\n".join([f"- {chunk['content']}" for chunk in retrieved_chunks])

    prompt = f"""You are a very capable person in turkish education system. Some questions will be asked to you in Turkish.
    You need to answer them in Turkish. For your convinience some context that will be helpful when answering is given to you.
    One important thing when answering is that you shouldn't answer the question with directly using the context. You should
    paraphrase and generate a new text when answering but don't go too far and generate unrelated or made up things.
    Just answer the question. Don't add anything else to your answer please. You can't copy the context directly. 
    Use your own sentences. Don't copy the prompt!

    ### Context is:
    {context}

    ### Question is:
    {query}

    
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text


def last_chance(query, retrieved_chunks, MODEL_NAME, tokenizer, model):
    
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    context = "\n\n".join([f"- {chunk['content']}" for chunk in retrieved_chunks])

    prompt = f"""Answer the user's query strictly using the provided retrieved documents. If the answer is not found in the retrieved 
    data, respond with: 'I could not find relevant information in the provided sources.' Do not generate any information beyond what 
    is retrieved. If the sources are contradictory, highlight the discrepancy without assuming or fabricating details. An example 
    generated text is like this:

    Example Question:
    Millî Eğitim Bakanlığı Öğretmen Atama ve Yer Değiştirme Yönetmeliğine göre, öğretmenlik görevi dışında bir göreve atanmış 
    olan öğretmenler tekrar öğretmenliğe dönebilir mi?

    Example Answer:
    Evet, bu yönetmeliğin yayımı tarihinden önce Bakanlığın merkez veya taşra teşkilatında eğitim ve öğretim hizmetleri sınıfı 
    dışındaki görevlere isteğiyle atanmış öğretmenler, yönetmeliğin yayımı tarihinden itibaren altı ay içinde başvuruda bulunmaları 
    hâlinde yeniden öğretmenliğe atanabilir.

    ### The context I want you to use is:
    {context}

    ### Question you need to answer is:
    {query}

    
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text



def verify_with_llm(answer, retrieved_chunks, query, MODEL_NAME, verifier_tokenizer, verifier_model):
    
    #VERIFIER_MODEL_NAME = "microsoft/phi-2"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # verifier_tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)
    # verifier_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    context = "\n\n".join([f"- {chunk['content']}" for chunk in retrieved_chunks])

    verification_prompt = f"""
    You will analyze whether a generated answer is supported by a given context.
    The answer **does not need to match word-for-word**, but it **must express the same meaning** as the information in the context.
    You must answer with only with the words "Supported" or "Not Supported". Don't write or generate any code. Don't copy from Context.
    Just write 'Supported' or 'Not Supported' under Your Answer should go here

    ### Context:
    {context}

    ### Question:
    {query}

    ### Generated Answer:
    {answer}

    ### Your Answer should go here:

    

    """

    inputs = verifier_tokenizer(verification_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = verifier_model.generate(**inputs, max_new_tokens=200)

    verification_result = verifier_tokenizer.decode(output[0], skip_special_tokens=True)
    return verification_result
# instructor bak
def answer_check(query, answer, tokenizer, model):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        

    prompt = f"""You are a very capable person in turkish education system. A question and it's answer will be given to you.
    What I want from you is grade the answer in terms of how well it answers the question given. The grading must be done like this:
    Assign a number to the answer where the number should be one of these (1,2,3,4,5). If the answer directly satisfies the question
    anwser with 5. If the answer isn't about the question answer with 1. Don't write any code. Just assign a number. Answer with a 
    numeric value. Don't copy this propmt!


    ### Question is:
    {query}

    ### Answer is:
    {answer}

    ### Your Answer should go here:

    
    """

    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(**inputs, max_new_tokens=200)
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

    return generated_text




def generation_pipeline(query, retrieved_chunks):
    """Full RAG pipeline with hallucination verification."""
    print("\n Generating answer...")

    MODEL_NAME = "microsoft/phi-2"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    answer = generate_answer(query, retrieved_chunks, MODEL_NAME, tokenizer, model)

    print("\n Verifying answer with LLM...")
    verification_result = verify_with_llm(answer, retrieved_chunks, query, MODEL_NAME, tokenizer, model)

    # Check verification result
    if "Not Supported" in verification_result:
        print("\n Hallucination detected! Retrying retrieval & regeneration...")
        #print(verification_result)
        second_answer = last_chance(query, retrieved_chunks, MODEL_NAME, tokenizer, model)

        check = answer_check(query, answer, tokenizer, model)
        
        return second_answer, check


    check = answer_check(query, answer, tokenizer, model)

    
    return answer, check



if __name__ == "__main__":
    test_query = "Sözleşmeli öğretmenliğe başvuru şartları nelerdir?"
    test_chunks = [{"content": "Başvuru için en az 50 KPSS puanı gereklidir."}] 
    
    final_answer, score = generation_pipeline(test_query, test_chunks)
    print("\n Generated Answer:\n", final_answer)
    print("\n Generated Answer:\n", score)