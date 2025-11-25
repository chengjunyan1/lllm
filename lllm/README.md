# Low-Level Languege Models (LLLM)  


Providing minimal necessary tools needed for building LLM agents:
1. Logging 
2. Dialog context management & Prunning
3. Prompt management & Reuse
4. Dialog exception handling
5. *Agent call*


Phylosophical goals:
- Keep it very simple, less layers of abstraction
- Keep it very flexible, no opinionated design
- Keep it very fast, no unnecessary overhead



## Definitions


### LLM

A function that maps embeddings of text, image, etc., into a string output. 

#### LLM Call

Given the dialog, return the response, and parse it based on the prompt on the top of the dialog.

### Agent Call

Maps the dialog and prompt arg, takes the Prompt function (see below), and return the expected output state and updated dialog.
Compared to LLM Call, it has the following differences:
 - It manages the dialog, it starts by loading the prompt into the dialog, and update it throughout the call.
 - It is stateful, and runs in a FSM way to ensure mapping from the initial state to the expected output state, or raise an error. 


### Prompt 

Prompt is a LLM wrapper with handlers and parsers that generating function of LLMs.
It contains a string used to call LLM --- a self-contained function with string arguments and dialog context, and the **handlers** that ensure it reaches the expected output string, as well as the parsers that convert the output string into a structured format (e.g., through XML tags).

#### Handlers

The program that handle all the states where the expected output is not reached. 
There are two types of handlers corresponding to the reasons of unreaching (due to error or not):

- Exception handlers: 
  - The expected output not reached due to an error in the response, notice that it does not catch the server side errors such as timeout, etc.
  - The dialogs triggered by exception handlings will be pruned.
- Interrupt handlers: 
  - The expected output not reached due to an interruption in the response, it is mainly the function calling.
  - The dialogs triggered by interrupt handlings will be kept for further processing.

The goal of the Prompt is to make it a more safe function, that ensuring the expected output is reached, unless it cannot be fixed with the maximum number of retries, or server side errors such as timeout, etc.

#### Function tool

Maps a Function Call to a string.
