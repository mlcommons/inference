#!/usr/bin/env python3
"""
Multi-Shot Prompting with OpenAI Harmony Format

This example demonstrates how to construct and tokenize a zero/multi-shot prompt using the
openai-harmony Python package for math500, aime1983, livecodebench, mmlu, gpqa with gpt-oss model series.
"""

import argparse
import json
import os
import pandas as pd
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Conversation,
    SystemContent,
    DeveloperContent,
    ReasoningEffort
)


def create_math500_prompt(user_query, reasoning_effort=ReasoningEffort.HIGH):
    """
    Creates a multi-shot prompt for mathematical problem solving using Harmony format.

    Returns:
        tuple: (conversation_object, token_list) ready for model completion
    """

    # Load the Harmony encoding for gpt-oss models
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Create the system message with required channels
    system_message = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date("2025-09-30")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    # Create the developer message with instructions
    developer_message = (
        DeveloperContent.new()
        .with_instructions(
            "You are a math expert that solves problems step-by-step. "
            "Always show your work clearly and put your final answer in \\boxed{answer} format. "
            "Follow the format shown in the examples below."
        )
    )

    # Create the conversation with multi-shot examples
    messages = [
        # System and developer setup
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),

        # Example 1: Square areas and side lengths
        Message.from_role_and_content(
            Role.USER,
            "The areas of three squares are 16, 49 and 169. What is the average (mean) of their side lengths?"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Since the areas of the three squares are 16, 49 and 169, then their side lengths are $\\sqrt{16}=4$, $\\sqrt{49}=7$ and $\\sqrt{169}=13$, respectively.\\n\\nThus, the average of their side lengths is $$\\frac{4+7+13}{3}=\\boxed{8}.$$"
        ).with_channel("final"),

        # Example 2: Floor function equation
        Message.from_role_and_content(
            Role.USER,
            "Find all $x$ such that $\\lfloor \\lfloor 2x \\rfloor - 1/2 \\rfloor = \\lfloor x + 2 \\rfloor.$"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Observe that $\\lfloor 2x \\rfloor$ is an integer, so it follows that $\\lfloor \\lfloor 2x \\rfloor - 1/2 \\rfloor = \\lfloor 2x \\rfloor - 1$. Also, $\\lfloor x + 2 \\rfloor = \\lfloor x \\rfloor + 2$. Thus, our equation becomes $$\\lfloor 2x \\rfloor = \\lfloor x \\rfloor + 3.$$Let $n = \\lfloor x \\rfloor,$ so $n \\le x < n + 1.$\\n\\nIf $x < n + \\frac{1}{2},$ then $2n \\le x < 2n + 1,$ so $\\lfloor 2x \\rfloor = 2n,$ and\\n\\[2n = n + 3,\\]which means $n = 3.$\\n\\nIf $x \\ge n + \\frac{1}{2},$ then $2n + 1 \\le x < 2n + 2,$ so $\\lfloor 2x \\rfloor = 2n + 1,$ and\\n\\[2n + 1 = n + 3,\\]which means $n = 2.$\\n\\nTherefore, the set of solutions is $x \\in \\boxed{\\left[ \\frac{5}{2}, \\frac{7}{2} \\right)}.$"
        ).with_channel("final"),

        # Example 3: Sequences and differences
        Message.from_role_and_content(
            Role.USER,
            "Sequence $A$ is a geometric sequence. Sequence $B$ is an arithmetic sequence. Each sequence stops as soon as one of its terms is greater than $300.$ What is the least positive difference between a number selected from sequence $A$ and a number selected from sequence $B?$\\n\\n$\\bullet$ Sequence $A:$ $2,$ $4,$ $8,$ $16,$ $32,$ $\\ldots$\\n\\n$\\bullet$ Sequence $B:$ $20,$ $40,$ $60,$ $80,$ $100,$ $\\ldots$"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "The terms of sequence $A$ are $2,$ $4,$ $8,$ $16,$ $32,$ $64,$ $128,$ $256,$ $512.$ The terms of sequence $B$ start from $20$ and go up by $20$ each time, so sequence $B$ is precisely all multiples of $20$ from $20$ to $320.$ We thus need to see which term in sequence $A$ is closest to a multiple of $20.$ $16,$ $64,$ and $256$ are the closest, each being $4$ away from a multiple of $20.$ So the least positive difference between a term in sequence $A$ and one in sequence $B$ is $\\boxed{4}.$"
        ).with_channel("final"),

        # Example 4: Probability and Deal or No Deal
        Message.from_role_and_content(
            Role.USER,
            "In the game Deal or No Deal, participants choose a box at random from a set of $26,$ one containing each of the following values: \\begin{tabular}{|c|c|}\\hline\\$.01&\\$1,000\\\\\\hline\\$1&\\$5,000\\\\\\hline\\$5&\\$10,000\\\\\\hline\\$10&\\$25,000\\\\\\hline\\$25&\\$50,000\\\\\\hline\\$50&\\$75,000\\\\\\hline\\$75&\\$100,000\\\\\\hline\\$100&\\$200,000\\\\\\hline\\$200&\\$300,000\\\\\\hline\\$300&\\$400,000\\\\\\hline\\$400&\\$500,000\\\\\\hline\\$500&\\$750,000\\\\\\hline\\$750&\\$1,000,000\\\\\\hline\\end{tabular} After choosing a box, participants eliminate other boxes by opening them, showing the amount of money in the box to the crowd, and then removing that box (and its money!) from the game. What is the minimum number of boxes a participant needs to eliminate in order to have a half chance of holding at least $\\$100,\\!000$ as his or her chosen box?"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Seven of the boxes contain at least this amount. If a participant is going to be holding one of these boxes with a probability of $1/2,$ there can be at most $7$ other boxes left. This means that at least $26-7-7=\\boxed{12}$ boxes must be eliminated."
        ).with_channel("final"),

        # Example 5: Domain of composite function
        Message.from_role_and_content(
            Role.USER,
            "Find the domain of the function $f(x) = \\tan(\\arccos(x^2)).$"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "For $\\arccos (x^2)$ to be defined, we must have $-1 \\le x^2 \\le 1,$ which is satisfied only for $-1 \\le x \\le 1.$  Then $\\arccos (x^2)$ will always return an angle between 0 and $\\frac{\\pi}{2}.$  Then $\\tan (\\arccos(x^2))$ is defined, unless $\\arccos(x^2) = \\frac{\\pi}{2}.$  This occurs only when $x = 0.$\\n\\nTherefore, the domain of $f(x)$ is $\\boxed{[-1,0) \\cup (0,1]}.$"
        ).with_channel("final"),

        # The actual problem to solve
        Message.from_role_and_content(
            Role.USER,
            user_query,
        )
    ]

    # Create the conversation
    convo = Conversation.from_messages(messages)

    # Render the conversation for completion (ready to send to the model)
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

    return convo, tokens


def create_aime1983_prompt(user_query, reasoning_effort=ReasoningEffort.HIGH):
    """
    Creates a zero-shot prompt for mathematical problem solving using Harmony format.

    Returns:
        tuple: (conversation_object, token_list) ready for model completion
    """

    # Load the Harmony encoding for gpt-oss models
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Create the system message with required channels
    system_message = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date("2025-09-30")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    # Create the developer message with instructions
    developer_message = (
        DeveloperContent.new()
        .with_instructions(
            "You are a math expert that solves problems step-by-step. "
            "The final line of your response should contain the final answer as an integer enclosed in \\boxed{answer}."
        )
    )

    messages = [
        # System and developer setup
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, user_query),
    ]

    # Create the conversation
    convo = Conversation.from_messages(messages)

    # Render the conversation for completion (ready to send to the model)
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

    return convo, tokens


def create_livecodebench_prompt(user_query, reasoning_effort=ReasoningEffort.HIGH):
    """
    Creates a zero-shot prompt for livecodebench problem solving using Harmony format.

    Returns:
        tuple: (conversation_object, token_list) ready for model completion
    """

    # Load the Harmony encoding for gpt-oss models
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Create the system message with required channels
    system_message = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date("2025-09-30")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    # Create the developer message with instructions
    developer_message = (
        DeveloperContent.new()
        .with_instructions(
            "You are a python coding expert that solves problems step-by-step. "
            "You must provide the reasoning to arriving at your solution and the code to solve the problem."
            "The code should be enclosed within ```python delimiters."
        )
    )

    messages = [
        # System and developer setup
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),
        Message.from_role_and_content(Role.USER, user_query),
    ]

    convo = Conversation.from_messages(messages)

    # Render the conversation for completion (ready to send to the model)
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

    return convo, tokens


def create_mmlu_prompt(user_query, reasoning_effort=ReasoningEffort.HIGH):
    """
    Creates a multi-shot prompt for multiple choice question answering using Harmony format.

    Returns:
        tuple: (conversation_object, token_list) ready for model completion
    """

    # Load the Harmony encoding for gpt-oss models
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Create the system message with required channels
    system_message = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date("2025-09-30")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    # Create the developer message with instructions
    developer_message = (
        DeveloperContent.new()
        .with_instructions(
            "You are an expert test-taker that answers multiple choice questions accurately. "
            "Follow the format shown in the examples below. "
            "The last line of your response should be in the following format: 'Answer: A/B/C/...' (e.g. 'Answer: A')."
        )
    )

    # Create the conversation with multi-shot examples
    messages = [
        # System and developer setup
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),

        # Example 1: Abstract Algebra - Ring Theory
        Message.from_role_and_content(
            Role.USER,
            "The symmetric group $S_n$ has $n!$ elements, hence it is not true that $S_{10}$ has 10 elements.\\nFind the characteristic of the ring 2Z.\\nA) 0\\nB) 30\\nC) 3\\nD) 10\\nE) 12\\nF) 50\\nG) 2\\nH) 100\\nI) 20\\nJ) 5"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "The characteristic of a ring R is the smallest positive integer n such that n·1_R = 0, or 0 if no such positive integer exists.\\n\\nFor the ring 2Z (even integers under addition and multiplication), we need to find the smallest positive integer n such that n·2 = 0 in 2Z. However, since 2Z contains all even integers, and n·2 is always even for any positive integer n, we have n·2 ≠ 0 for any positive n in 2Z.\\n\\nTherefore, the characteristic of 2Z is 0.\\n\\nAnswer: A"
        ).with_channel("final"),

        # Example 2: Linear Algebra - Transformations
        Message.from_role_and_content(
            Role.USER,
            "Let V be the set of all real polynomials p(x). Let transformations T, S be defined on V by T:p(x) -> xp(x) and S:p(x) -> p'(x) = d/dx p(x), and interpret (ST)(p(x)) as S(T(p(x))). Which of the following is true?\\nA) ST + TS is the identity map of V onto itself.\\nB) TS = 0\\nC) ST = 1\\nD) ST - TS = 0\\nE) ST = T\\nF) ST = 0\\nG) ST = TS\\nH) ST - TS is the identity map of V onto itself.\\nI) TS = T\\nJ) ST = S"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Let's compute ST and TS for a general polynomial p(x).\\n\\nFor ST: (ST)(p(x)) = S(T(p(x))) = S(xp(x)) = d/dx(xp(x)) = p(x) + xp'(x)\\n\\nFor TS: (TS)(p(x)) = T(S(p(x))) = T(p'(x)) = xp'(x)\\n\\nNow let's find ST - TS:\\n(ST - TS)(p(x)) = (p(x) + xp'(x)) - xp'(x) = p(x)\\n\\nSince (ST - TS)(p(x)) = p(x) for any polynomial p(x), we have ST - TS = I, where I is the identity transformation.\\n\\nAnswer: H"
        ).with_channel("final"),

        # Example 3: Number Theory - Diophantine Equations
        Message.from_role_and_content(
            Role.USER,
            "Let A be the set of all ordered pairs of integers (m, n) such that 7m + 12n = 22. What is the greatest negative number in the set B = {m + n : (m, n) ∈ A}?\\nA) -5\\nB) 0\\nC) -3\\nD) -7\\nE) -4\\nF) -6\\nG) -1\\nH) -2\\nI) -9\\nJ) N/A"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "First, I need to find the general solution to 7m + 12n = 22.\\n\\nUsing the extended Euclidean algorithm: gcd(7,12) = 1, so solutions exist.\\nA particular solution is m₀ = 10, n₀ = -4 (since 7(10) + 12(-4) = 70 - 48 = 22).\\n\\nThe general solution is:\\nm = 10 + 12t\\nn = -4 - 7t\\nfor integer t.\\n\\nSo m + n = (10 + 12t) + (-4 - 7t) = 6 + 5t\\n\\nFor different values of t:\\nt = 0: m + n = 6\\nt = -1: m + n = 1\\nt = -2: m + n = -4\\nt = -3: m + n = -9\\nt = 1: m + n = 11\\n\\nThe greatest negative number is -4.\\n\\nAnswer: E"
        ).with_channel("final"),

        # Example 4: Differential Equations - Salt Tank Problem
        Message.from_role_and_content(
            Role.USER,
            "A tank initially contains a salt solution of 3 grams of salt dissolved in 100 liters of water. A salt solution containing 0.02 grams of salt per liter of water is sprayed into the tank at a rate of 4 liters per minute. The sprayed solution is continually mixed with the salt solution in the tank, and the mixture flows out of the tank at a rate of 4 liters per minute. If the mixing is instantaneous, how many grams of salt are in the tank after 100 minutes have elapsed?\\nA) 3 + e^-2\\nB) 2 - e^-4\\nC) 2 - e^-2\\nD) 3 + e^-4\\nE) 2 + e^-3\\nF) 2 - e^-3\\nG) 3 - e^-2\\nH) 2 + e^-2\\nI) 2 + e^-4\\nJ) 2"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Let S(t) be the amount of salt at time t. The tank volume remains constant at 100L since inflow = outflow = 4 L/min.\\n\\nSalt enters at rate: 4 L/min × 0.02 g/L = 0.08 g/min\\nSalt leaves at rate: 4 L/min × S(t)/100 g/L = 0.04S(t) g/min\\n\\nDifferential equation: dS/dt = 0.08 - 0.04S(t)\\n\\nRearranging: dS/dt + 0.04S = 0.08\\n\\nThis is first-order linear. The integrating factor is e^(0.04t).\\n\\nSolution: S(t) = 2 + Ce^(-0.04t)\\n\\nUsing initial condition S(0) = 3:\\n3 = 2 + C, so C = 1\\n\\nTherefore: S(t) = 2 + e^(-0.04t)\\n\\nAt t = 100: S(100) = 2 + e^(-4) = 2 + e^-4\\n\\nAnswer: I"
        ).with_channel("final"),

        # Example 5: Basic Arithmetic - Division
        Message.from_role_and_content(
            Role.USER,
            "A total of 30 players will play basketball at a park. There will be exactly 5 players on each team. Which statement correctly explains how to find the number of teams needed?\\nA) Multiply 5 by 5 to find 25 teams.\\nB) Divide 30 by 5 to find 6 teams.\\nC) Add 5 to 30 to find 35 teams.\\nD) Subtract 30 from 5 to find -25 teams.\\nE) Divide 5 by 30 to find 0.1667 teams.\\nF) Add 5 to 30 then divide by 2 to find 17.5 teams.\\nG) N/A\\nH) N/A\\nI) N/A\\nJ) N/A"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "To find the number of teams when we know the total number of players and the number of players per team, we need to divide the total number of players by the number of players per team.\\n\\nTotal players = 30\\nPlayers per team = 5\\n\\nNumber of teams = Total players ÷ Players per team = 30 ÷ 5 = 6\\n\\nTherefore, we divide 30 by 5 to find 6 teams.\\n\\nAnswer: B"
        ).with_channel("final"),

        # The actual problem to solve
        Message.from_role_and_content(
            Role.USER,
            user_query,
        )
    ]

    # Create the conversation
    convo = Conversation.from_messages(messages)

    # Render the conversation for completion (ready to send to the model)
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

    return convo, tokens


def create_gpqa_prompt(user_query, reasoning_effort=ReasoningEffort.HIGH):
    """
    Creates a multi-shot prompt for organic chemistry and biochemistry question answering using Harmony format.

    Returns:
        tuple: (conversation_object, token_list) ready for model completion
    """

    # Load the Harmony encoding for gpt-oss models
    enc = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    # Create the system message with required channels
    system_message = (
        SystemContent.new()
        .with_reasoning_effort(reasoning_effort)
        .with_conversation_start_date("2025-09-30")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    # Create the developer message with instructions
    developer_message = (
        DeveloperContent.new()
        .with_instructions(
            "You are an expert in organic chemistry and biochemistry who answers scientific questions accurately. "
            "Provide detailed explanations showing your reasoning, then conclude with the answer. "
            "The last line of your response should be in the following format: 'Answer: A/B/C/D' (e.g. 'Answer: A')."
        )
    )

    # Create the conversation with multi-shot examples
    messages = [
        # System and developer setup
        Message.from_role_and_content(Role.SYSTEM, system_message),
        Message.from_role_and_content(Role.DEVELOPER, developer_message),

        # Example 1: Molecular Biology - Gene Therapy
        Message.from_role_and_content(
            Role.USER,
            "A large gene has dozens of exons, of which the central ones code for folded triple helical repeats that connect the cytoskeleton with sarcolemma and extracellular space. Each exon usually codes for one folded triple alpha helix. The most common mutations of the gene are central exon deletions that create out-of-frame peptides and progressive degenerative organ waste. A solution is to deliver a Morpholino that recognizes the 5' end of the out-of-frame exon in pre-mRNA. The molecule prevents binding of the spliceosome and creates exon skipping and in-frame joining. Several missing exons are well tolerated by an organism. Which structure below is not involved in the proposed therapy?\\nA) polyA tail\\nB) lariat\\nC) antisense\\nD) R-loops"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Explanation: The text describes the dystrophin gene and the FDA-approved oligonucleotide therapy that causes exon skipping by creating a functional, albeit shorter, dystrophin protein. Let me analyze each option:\\n\\n- Morpholino is bound to the pre-mRNA in an antisense orientation (C is involved)\\n- Every splicing mechanism creates the lariat molecule that is circular with a 3' tail and soon degraded (B is involved)\\n- The spliced RNA is polyadenylated at the 3' end with a polyA tail (A is involved)\\n- R-loops are triple helix structures of DNA and pre-mRNA that are a consequence of RNA transcription, not splicing and RNA maturation\\n\\nR-loops are not involved in the proposed exon-skipping therapy mechanism.\\n\\nAnswer: D"
        ).with_channel("final"),

        # Example 2: Stereochemistry - Optical Activity
        Message.from_role_and_content(
            Role.USER,
            "How many of the following compounds exhibit optical activity?\\n1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene\\n2,3,3,3-tetrafluoroprop-1-ene\\ndi(cyclohex-2-en-1-ylidene)methane\\n5-(5-methylhexan-2-ylidene)cyclopenta-1,3-diene\\n3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene\\n[1,1'-biphenyl]-3,3'-diol\\n8,8-dichlorobicyclo[4.2.0]octan-7-one\\ncyclopent-2-en-1-one\\nA) 6\\nB) 5\\nC) 4\\nD) 3"
        ),
        Message.from_role_and_content(
            Role.ASSISTANT,
            "Explanation: To exhibit optical activity, a compound must be chiral (lack a mirror plane of symmetry). Let me analyze each compound:\\n\\nOptically active (chiral) compounds:\\n- 1-methyl-4-(prop-1-en-2-yl)cyclohex-1-ene: has chiral center\\n- di(cyclohex-2-en-1-ylidene)methane: no plane of symmetry\\n- 3-(2-methylbut-1-en-1-ylidene)cyclohex-1-ene: chiral\\n- 8,8-dichlorobicyclo[4.2.0]octan-7-one: chiral bicyclic structure\\n\\nOptically inactive (achiral) compounds:\\n- 2,3,3,3-tetrafluoroprop-1-ene: has mirror plane\\n- 5-(5-methylhexan-2-ylidene)cyclopenta-1,3-diene: has mirror plane\\n- [1,1'-biphenyl]-3,3'-diol: has mirror plane\\n- cyclopent-2-en-1-one: has mirror plane\\n\\nTherefore, 4 compounds exhibit optical activity.\\n\\nAnswer: C"
        ).with_channel("final"),

        # The actual problem to solve
        Message.from_role_and_content(
            Role.USER,
            user_query,
        )
    ]

    # Create the conversation
    convo = Conversation.from_messages(messages)

    # Render the conversation for completion (ready to send to the model)
    tokens = enc.render_conversation_for_completion(convo, Role.ASSISTANT)

    return convo, tokens


def process_row(args):
    """
    Worker function to process a single row from the dataframe.

    Args:
        args: tuple of (index, row, dataset_function_map)

    Returns:
        tuple: (index, convo, tokens, dataset_name) or (index, None, None, dataset_name, error)
    """
    index, row, dataset_function_map = args
    dataset_name = row["dataset"]
    user_query = row["question"]

    try:
        # Get the appropriate function based on dataset type
        if dataset_name in dataset_function_map:
            create_prompt_func = dataset_function_map[dataset_name]
            convo, tokens = create_prompt_func(user_query)
            return (index, convo, tokens, dataset_name)
        else:
            error_msg = f"Unknown dataset '{dataset_name}' at index {index}"
            return (index, None, None, dataset_name, error_msg)
    except Exception as e:
        error_msg = f"Error processing {dataset_name} at index {index}: {str(e)}"
        return (index, None, None, dataset_name, error_msg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-file",
        type=str,
        default="/home/mlperf_inference_storage/data/deepseek-r1/mlperf_deepseek_r1_dataset_4388_fp8_eval.pkl")
    parser.add_argument("--num-processes", type=int, default=cpu_count(),
                        help="Number of processes to use for parallel processing (default: number of CPU cores)")
    parser.add_argument("--max-rows", type=int, default=None,
                        help="Maximum number of rows to process (default: process all rows)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output pickle file path to save the processed data as pandas DataFrame")
    parser.add_argument("--reasoning-effort", type=str, default="high",
                        help="Reasoning effort to use for the prompt")
    args = parser.parse_args()
    df = pd.read_pickle(args.data_file)

    reasoning_effort = {
        "high": ReasoningEffort.HIGH,
        "medium": ReasoningEffort.MEDIUM,
        "low": ReasoningEffort.LOW
    }[args.reasoning_effort.lower()]

    # Apply row limit if specified
    if args.max_rows is not None:
        df = df.head(args.max_rows)
        print(f"Limited to first {args.max_rows} rows")

    # Create mapping from dataset names to prompt creation functions
    dataset_function_map = {
        'aime1983': create_aime1983_prompt,
        'gpqa': create_gpqa_prompt,
        'livecodebench': create_livecodebench_prompt,
        'math500': create_math500_prompt,
        'mmlu_pro': create_mmlu_prompt
    }

    # Prepare data for parallel processing
    process_args = [(index, row, dataset_function_map)
                    for index, row in df.iterrows()]

    # Don't use more processes than we have rows
    num_processes = min(args.num_processes, len(df))

    print(f"Processing {len(df)} queries using {num_processes} processes...")

    # Process rows in parallel with progress bar
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_row, process_args),
            total=len(process_args),
            desc="Processing queries",
            unit="query"
        ))

    # Sort results by index to preserve original order
    results.sort(key=lambda x: x[0])

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process results and modify original DataFrame
    successful_count = 0
    error_count = 0

    # Set reference columns to None
    df['ref_accuracy'] = None
    df['ref_extracted_answer'] = None
    df['ref_output'] = None
    df['tok_ref_output'] = None
    df['tok_ref_output_len'] = None

    # Set templated text input to None (if it exists)
    if 'templated_text_input' in df.columns:
        df['templated_text_input'] = None

    # Initialize columns for harmony tokenized input
    df['tok_input'] = None
    df['tok_input_len'] = None
    df['text_input'] = None

    for result in tqdm(results, desc="Processing results", unit="entry"):
        if len(result) == 5:  # Error case
            index, convo, tokens, dataset_name, error_msg = result
            error_count += 1

            # Update the original DataFrame with error data
            df.at[index, 'tok_input'] = None
            df.at[index, 'tok_input_len'] = None
            df.at[index, 'text_input'] = None
        else:  # Success case
            index, convo, tokens, dataset_name = result
            successful_count += 1

            # Convert conversation to string format
            conversation_parts = []
            for message in convo.messages:
                # Get role from message.author.role
                role = message.author.role.value if hasattr(
                    message.author.role, 'value') else str(
                    message.author.role)

                # Get content from message.content (which is a list)
                content_parts = []
                for content_item in message.content:
                    if hasattr(content_item, 'text'):
                        content_parts.append(content_item.text)
                    else:
                        content_parts.append(str(content_item))
                content = ' '.join(content_parts)

                # Format as "Role: content"
                conversation_parts.append(f"{role}: {content}")

            conversation_string = '\n'.join(conversation_parts)

            # Update the original DataFrame with successful data
            df.at[index, 'tok_input'] = tokens.tolist() if hasattr(
                tokens, 'tolist') else list(tokens)
            df.at[index, 'tok_input_len'] = len(
                tokens) if hasattr(tokens, '__len__') else 0
            df.at[index, 'text_input'] = conversation_string

    # Verify input and output have identical column lists
    input_columns = list(pd.read_pickle(args.data_file).columns)
    output_columns = list(df.columns)
    if input_columns != output_columns:
        print(f"WARNING: Column lists differ!")
        print(f"Input columns: {input_columns}")
        print(f"Output columns: {output_columns}")

    # Save the modified DataFrame as pickle
    print("Saving modified DataFrame to pickle...")
    df.to_pickle(args.output_file)

    print(f"\nProcessing completed:")
    print(f"  - Successfully processed: {successful_count} queries")
    print(f"  - Errors: {error_count} queries")
    print(f"  - Output saved to: {args.output_file} (pickle format)")
    print(f"  - Total processed: {successful_count + error_count} queries")
    print(f"  - Modified DataFrame shape: {df.shape}")
    print(f"  - Updated columns: tok_input, tok_input_len, text_input")
    print(f"  - Set to None: ref_accuracy, ref_extracted_answer, ref_output, tok_ref_output, tok_ref_output_len, templated_text_input")
