from chatbot.dataprep import DataPreProcessor
from chatbot.ragbot import RagPipeline
from transformers import pipeline
from openai import OpenAI
import configparser

## turned off for now!
# from chatbot.finetuned_slm import load_model, predict

class CustomGuardrail:
    def __init__(self, config_path='config.env'):
        self._load_config(config_path)
        self._initialize_chatbot()
        self.verbose = True

    def _load_config(self, config_path):
        config = configparser.ConfigParser()
        config.read(config_path)
        
        self.allowed_topics = config['DEFAULTS']['ALLOWED_TOPICS'].split()
        self.default_response = config['DEFAULTS']['DEFAULT_PROMPT']
        self.model_name = config['OPENAI']['MODELNAME']
        self.api_key = config['OPENAI']['APIKEY']
        self.client = OpenAI(api_key=self.api_key)

        self.UNETHICAL_KEYWORDS = [
            "hate", "violence", "racist", "sexist", "abuse", "threat", "kill", "murder", "attack",
            "harassment", "fraud", "scam", "bully", "torture", "suicide", "self harm", "exploit",
            "terror", "illegal", "drugs", "weapon", "bomb", "extortion", "blackmail", "male", "female"
        ]

    def _initialize_chatbot(self):
        dataprep = DataPreProcessor()
        vector_store = dataprep.load_vectordb()
        self.chatbot = RagPipeline(vector_store, self.model_name, self.api_key)

    def _log(self, message):
        if self.verbose:
            print(message)

    def intent_guardrail(self, prompt):
        ###### 
        ## option1 : Fine tuned SLM 
        # used my fine tuned model ie. bank_finetuned_model but initial results were not good, 
        # hence used a zero shot classification model! 

        # 
        # text_samples = [prompt]
        # model, tokenizer = load_model()
        # predict(model, tokenizer, text_samples)

        ######

        ## option2: use a standard zero shot classifier
        labels = self.allowed_topics + ["offtopic"]
        classifier = pipeline("zero-shot-classification", model="cross-encoder/nli-MiniLM2-L6-H768")
        response = classifier(prompt, candidate_labels=labels)
        return response["labels"][0].lower() != "offtopic"

    def detect_unethical_content(self, text):
        return any(keyword in text.lower() for keyword in self.UNETHICAL_KEYWORDS)

    def reasoning_llm_response(self, query, context, response):
        reasoning_prompt = f"""
        Instructions:
            - Think step by step and carefully analyze the components: user query, context, and response.
            - Ensure the response is strictly based on the given context.
            - Provide a score between 1 and 10, where 1 means it is a response within the context and 10 means it is not ie. probably hallucination.
        
        Text to Validate:
            <START OF QUERY> {query} <END OF QUERY>  
            <START OF CONTEXT> {context} <END OF CONTEXT>   
            <START OF RESPONSE> {response} <END OF RESPONSE>   
        
        ### JUST REPLY WITH A SCORE - one number only!
        """

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": reasoning_prompt}],
            temperature=0
        )

        return int(response.choices[0].message.content)

    def execute_with_guardrail(self, prompt):
        self._log(f"User Prompt: {prompt}")
        guardrail_flag = True
        response, message = "", ""

        self._log("Running Guardrail 1: Prompt Intent Validation")
        if not self.intent_guardrail(prompt) or self.detect_unethical_content(prompt):
            guardrail_flag = False
            message = "❌ Prompt rejected: Not allowed."
            response = self.default_response

        if not guardrail_flag:
            return self._generate_result(guardrail_flag, response, message)
        
        self._log("Running RAG Pipeline with Strict Prompting")
        prompt, context, response = self.chatbot.execute(prompt)
        if "sorry" in response.lower()[:200]:
            message = "❌ Information not found!"
            guardrail_flag = False

        if not guardrail_flag:
            return self._generate_result(guardrail_flag, response, message)

        self._log("Running Guardrail 3: Reasoning the Response")
        reasoning_score = self.reasoning_llm_response(prompt, context, response)
        
        if reasoning_score > 5:
            guardrail_flag = False
            message = f"❌ Reasoning guardrail triggered. Score: {reasoning_score}"
            response = self.default_response
        else:
            message = "✅ Response passed reasoning"

        return self._generate_result(guardrail_flag, response, message)

    def _generate_result(self, flag, response, message):
        return {
            "guardrail_flag": flag,
            "response": response,
            "message": message
        }
