import chainlit as cl
import asyncio
from chatbot.guardrail import CustomGuardrail

guardrail = CustomGuardrail(config_path='config.env')

@cl.on_message
async def main(message: cl.Message):
    output = guardrail.execute_with_guardrail(message.content)
    response = output['response']
    message_content = output['message']

    # Guardrails Message
    with cl.Step(name="Guardrails: " + message_content):
        await cl.Message(content="").send()

    # Response
    msg = cl.Message(content="")  
    await msg.send()

    for line in response.split("\n"):
        for word in line.split():
            await asyncio.sleep(0.1)  
            await msg.stream_token(word + " ")  
        await msg.stream_token("\n")  
    await msg.update()  
