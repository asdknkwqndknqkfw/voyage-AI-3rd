import json
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


# 답변을 위한 Pydantic 모델 정의
class ProblemAnswer(BaseModel):
    problem_id: str = Field(description="문제 ID")
    answer: int = Field(description="1부터 5까지의 답안 번호")
    explanation: str = Field(description="답을 선택한 이유에 대한 설명")


class KICE_ANSWER(BaseModel):
    answers: List[ProblemAnswer] = Field(description="각 문제에 대한 답변 목록")


# 입력 JSON 파일에서 데이터 로드
def load_data(file_path: str) -> List[Dict[str, Any]]:
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


# 프롬프트 템플릿 정의
prompt_template = """
당신은 한국어 독해 문제를 푸는 전문가입니다. 주어진 지문과 문제를 분석하고 가장 적절한 답을 선택해주세요.

### 지문:
{paragraph}

### 문제 {problem_index}: 
{question}
{question_plus}

### 선택지:
{choices}

다음 형식으로 답변해주세요:
1. 각 선택지를 분석하여 지문 내용과 일치하는지 확인
2. 가장 적절한 답 선택하기 (1부터 5까지 번호로 표시)
3. 답을 선택한 이유 설명하기

지문을 철저히 분석하고, 지문에 명시된 내용을 바탕으로 답을 선택해주세요.
"""


def format_choices(choices: List[str]) -> str:
    return "\n".join([f"{i+1}. {choice}" for i, choice in enumerate(choices)])


def process_problems(data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []

    for item in data:
        if "paragraph" not in item or "problems" not in item:
            continue

        paragraph = item["paragraph"]
        problems = item["problems"]

        for i, problem in enumerate(problems):
            # 선택지 포맷팅
            choices_text = format_choices(problem["choices"])

            # 추가 질문이 있으면 포함
            question_plus = problem.get("question_plus", "")

            # 프롬프트 준비
            messages = [
                {
                    "role": "system",
                    "content": "당신은 한국어 독해 평가 전문가입니다. 주어진 지문과 문제를 분석하여 가장 적절한 답을 찾아주세요.",
                },
                {
                    "role": "user",
                    "content": prompt_template.format(
                        paragraph=paragraph,
                        problem_index=i + 1,
                        question=problem["question"],
                        question_plus=question_plus,
                        choices=choices_text,
                    ),
                },
            ]

            # 문제 ID 생성
            problem_id = f"{item.get('id', 'unknown')}_{i+1}"
            results.append(
                {
                    "problem_id": problem_id,
                    "messages": messages,
                    "expected_answer": problem.get("answer", None),
                }
            )

    return results


def solve_problems(data_file: str):
    # 데이터 로드 및 처리
    data = load_data(data_file)
    problems = process_problems(data)

    # LangChain 컴포넌트 설정
    output_parser = JsonOutputParser(pydantic_object=KICE_ANSWER)

    # OpenAI 모델 초기화
    model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    all_answers = []

    for problem in problems:
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 한국어 독해 평가 전문가입니다. 주어진 지문과 문제를 분석하여 가장 적절한 답을 찾아주세요.",
                ),
                (
                    "user",
                    """
### 지문:
{paragraph}

### 문제: 
{question}
{question_plus}

### 선택지:
{choices}

주어진 지문을 바탕으로 가장 적절한 답을 선택해주세요. 답은 1부터 5까지의 숫자로 표시하고, 왜 그 답을 선택했는지 간단히 설명해주세요.
""",
                ),
            ]
        )

        # Chain 구성
        chain = prompt | model | output_parser

        # 사용할 파라미터 추출
        problem_data = {
            "paragraph": data[0]["paragraph"],
            "question": problem["messages"][1]["content"]
            .split("### 문제")[1]
            .split("### 선택지")[0],
            "question_plus": "",
            "choices": problem["messages"][1]["content"].split("### 선택지:")[1],
        }

        # Chain 실행
        response = chain.invoke(problem_data)
        print(f"response: {response}")

        # 정답과 비교
        expected = problem.get("expected_answer")
        # actual = response.answers[0].answer
        actual = response

        all_answers.append(
            {
                "problem_id": problem["problem_id"],
                "predicted_answer": actual,
                "expected_answer": expected,
                "correct": actual == expected,
                "explanation": response.answers[0].explanation,
            }
        )

    return all_answers


# 메인 함수
def main():
    results = solve_problems("a.json")

    # 결과 출력
    correct_count = sum(1 for r in results if r["correct"])
    print(
        f"정확도: {correct_count}/{len(results)} ({correct_count/len(results)*100:.2f}%)"
    )

    for r in results:
        print(f"문제 ID: {r['problem_id']}")
        print(f"예측 답안: {r['predicted_answer']}")
        print(f"실제 답안: {r['expected_answer']}")
        print(f"정답 여부: {'O' if r['correct'] else 'X'}")
        print(f"설명: {r['explanation']}")
        print("-" * 50)


if __name__ == "__main__":
    main()
