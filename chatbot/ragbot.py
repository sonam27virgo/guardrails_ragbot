from openai import OpenAI

class RagPipeline:
    def __init__(self, vector_store, model_name, api_key, verbose=False):
        self.vector_store = vector_store
        self.client = OpenAI(api_key=api_key)
        self.model_name = model_name ## improvements here - other model choices
        self.verbose = verbose

    def retrieve(self, query):
        return self.vector_store.similarity_search(query)

    def generate(self, query, similar_chunks):

        context = "\n\n".join(doc.page_content for doc in similar_chunks)

        ## Strict Prompting Goes Here: 
        system_prompt = """You are a banking chatbot for The Bank, designed to provide accurate and relevant information strictly about retail banking products.
            Your primary role is to assist customers by answering questions about:
                - Credit Cards
                - Home Loans
                - Personal Loans
                - Savings Accounts
                - Term Deposits
                - Transaction Accounts

            Strict Guidelines:
                - Stay within scope – Only provide information about the bank’s retail products.
                - No personal advice – Do not offer financial, investment, or legal advice.
                - Direct users to official bank representatives for personalized assistance.
                - No speculation or opinions – Respond only with factual, product-related information based on verified bank data.
                - No external topics – Do not discuss non-banking topics, company policies, security issues, or regulatory compliance.
                - No confidential data – Never request, store, or process sensitive customer information (e.g., account details, passwords, personal identity data).
                - Accuracy and neutrality – Ensure responses are fact-based, clear, and neutral, without bias or promotional language.

            Security disclaimer –
                - If a user asks about security, fraud, or suspicious transactions, advise them to contact the bank directly via official channels.
                - If a user asks about fraud, phishing, or compromised accounts, immediately redirect them to official support.
            Answer the question in a factual and sincere tone.
        """
        
        pre_context = """
        Pay attention and remember the information below. Use strictly only the below context: """

        post_context = """
        According to the given information above, provide a well-structured response that begins with "According to", If you are unable to answer the query based on information provided, start your response with "Sorry: """

        full_prompt = f"""
        Instructions: {system_prompt}
        
        {pre_context} 

        Context: {context} 

        {post_context}
        
        Query: {query}
        """
 
        response = self.client.chat.completions.create(
            model = self.model_name,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": full_prompt}
            ]
        )
        return query, context, response.choices[0].message.content

    def execute(self, prompt):
        """Executes the retrieval and generation pipeline."""
        if self.verbose:
            print(f"Prompt: {prompt}")
            print("Preparing context chunks...")

        similar_chunks = self.retrieve(prompt)

        if self.verbose:
            print("Generating response...")

        prompt, context, response = self.generate(prompt, similar_chunks)
        return prompt, context, response