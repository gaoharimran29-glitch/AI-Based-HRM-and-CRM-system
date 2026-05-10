from pymongo import MongoClient
from langchain_core.tools import tool
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone
import os
from dotenv import load_dotenv
from policies import vectorstore

# -----------------------------
# DB CONNECTION
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["hr_ai_system"]


def _get_leave_balance(employee_id: str):
    record = db.leaves.find_one({"employee_id": employee_id})
    
    if not record:
        return {"status": "error", "message": "Leave record not found"}

    leave_balance = record.get("leave_balance", {})

    total = 0
    used = 0

    breakdown = {}

    for leave_type, data in leave_balance.items():
        t = data.get("total", 0)
        u = data.get("used", 0)
        cf = data.get("carry_forward", 0) if leave_type == "earned" else 0

        # Add carry forward to total (only for earned)
        effective_total = t + cf

        total += effective_total
        used += u

        breakdown[leave_type] = {
            "total": effective_total,
            "used": u,
            "remaining": effective_total - u
        }

    return {
        "status": "success",
        "total": total,
        "used": used,
        "remaining": total - used,
        "breakdown": breakdown,
        "pending_requests": record.get("pending_requests", 0),
        "last_leave_taken": record.get("last_leave_taken")
    }


def _get_salary(employee_id: str):
    record = db.salary.find_one({"employee_id": employee_id})

    if not record:
        return {"status": "error", "message": "Salary record not found"}

    monthly = record.get("monthly", {})
    deductions = record.get("deductions", {})

    # Calculate totals dynamically
    total_earnings = sum(monthly.values())
    total_deductions = sum(deductions.values())

    return {
        "status": "success",
        "ctc": record.get("ctc", 0),
        "earnings": monthly,
        "total_earnings": total_earnings,
        "deductions": deductions,
        "total_deductions": total_deductions,
        "net_salary": record.get("net_salary", total_earnings - total_deductions),
        "pay_grade": record.get("pay_grade"),
        "last_increment_date": record.get("last_increment_date"),
        "increment_percentage": record.get("increment_percentage")
    }

def _get_employee_info(employee_id: str):
    record = db.employees.find_one({"_id": employee_id})

    if not record:
        return {"status": "error", "message": "Employee not found"}

    return {
        "status": "success",
        "employee_code": record.get("employee_code"),
        "name": record.get("name"),
        "email": record.get("email"),
        "phone": record.get("phone"),
        "role": record.get("role"),
        "department": record.get("department"),
        "designation": record.get("designation"),
        "manager_id": record.get("manager_id"),
        "date_of_joining": record.get("date_of_joining"),
        "employment_type": record.get("employment_type"),
        "work_mode": record.get("work_mode"),
        "location": record.get("location"),
        "status": record.get("status"),
        "skills": record.get("skills", []),
        "performance_rating": record.get("performance_rating"),
        # ⚠️ Do NOT expose full bank details
        "bank_masked": record.get("bank_details", {}).get("account_number", "XXXX")
    }

def _query_policy_rag(query: str) -> str:
    docs = vectorstore.similarity_search(query, k=3)

    if not docs:
        return "No relevant policy found."

    context = "\n".join([doc.page_content for doc in docs])[:1500]

    return f"""Use the following company policy context to answer the user's question clearly:

    {context}

    Answer:"""

# =====================================================
# TOOL WRAPPER (LLM SAFE)
# =====================================================

def get_tools(employee_id: str):

    @tool
    def leave_balance_tool(employee_id: str) -> str:
        """
        Fetch the employee's leave balance including:
        - total leaves
        - used leaves
        - remaining leaves
        - leave type breakdown (casual, sick, earned)
        - pending leave requests
        - last leave taken

        MUST be used for:
        - leave balance queries
        - leave eligibility questions (e.g., "Can I take leave?")
        
        If the query also involves company rules,
        call this tool FIRST, then policy_rag_tool.
        """

        data = _get_leave_balance(employee_id)

        if data.get("status") == "error":
            return data["message"]

        breakdown_text = "\n".join([
            f"{k.capitalize()}: {v['remaining']} remaining (Used: {v['used']}/{v['total']})"
            for k, v in data["breakdown"].items()
        ])

        return (
            f"Leave Summary:\n"
            f"- Total: {data['total']}\n"
            f"- Used: {data['used']}\n"
            f"- Remaining: {data['remaining']}\n\n"
            f"- Breakdown:\n{breakdown_text}\n\n"
            f"- Pending Requests: {data['pending_requests']}\n"
            f"- Last Leave Taken: {data['last_leave_taken']}"
        )

    @tool
    def salary_tool(employee_id: str) -> str:
        """
        Fetch the employee's salary details including:
        - CTC
        - monthly earnings breakdown
        - deductions breakdown
        - net salary
        - pay grade
        - last increment info

        MUST be used for:
        - salary queries
        - salary breakdown
        - payslip explanation
        - compensation comparison

        If query involves company policy,
        call this tool FIRST, then policy_rag_tool.
        """

        data = _get_salary(employee_id)

        if data.get("status") == "error":
            return data["message"]

        earnings_text = "\n".join([
            f"{k.replace('_', ' ').title()}: ₹{v}"
            for k, v in data["earnings"].items()
        ])

        deductions_text = "\n".join([
            f"{k.upper()}: ₹{v}"
            for k, v in data["deductions"].items()
        ])

        return (
            f"Salary Summary:\n"
            f"- CTC: ₹{data['ctc']}\n"
            f"- Net Salary: ₹{data['net_salary']}\n\n"

            f"Earnings Breakdown:\n{earnings_text}\n"
            f"Total Earnings: ₹{data['total_earnings']}\n\n"

            f"Deductions:\n{deductions_text}\n"
            f"Total Deductions: ₹{data['total_deductions']}\n\n"

            f"Pay Grade: {data['pay_grade']}\n"
            f"Last Increment: {data['last_increment_date']} "
            f"({data['increment_percentage']}%)"
        )

    @tool
    def employee_info_tool(employee_id: str) -> str:
        """
        Fetch the employee's profile information including:
        - personal details
        - job role, department, designation
        - manager
        - skills & performance
        - employment details

        MUST be used for:
        - profile queries
        - role-based eligibility
        - promotion / appraisal queries

        If query involves company policy,
        call this tool FIRST, then policy_rag_tool.
        """

        data = _get_employee_info(employee_id)

        if data.get("status") == "error":
            return data["message"]

        skills_text = ", ".join(data["skills"]) if data["skills"] else "Not specified"

        return (
            f"👤 Employee Profile:\n"
            f"- Name: {data['name']}\n"
            f"- Employee Code: {data['employee_code']}\n"
            f"- Email: {data['email']}\n"
            f"- Phone: {data['phone']}\n\n"

            f"💼 Job Details:\n"
            f"- Role: {data['role']}\n"
            f"- Department: {data['department']}\n"
            f"- Designation: {data['designation']}\n"
            f"- Manager ID: {data['manager_id']}\n\n"

            f"📅 Employment Info:\n"
            f"- Joining Date: {data['date_of_joining']}\n"
            f"- Type: {data['employment_type']}\n"
            f"- Work Mode: {data['work_mode']}\n"
            f"- Location: {data['location']}\n"
            f"- Status: {data['status']}\n\n"

            f"🧠 Skills: {skills_text}\n"
            f"⭐ Performance Rating: {data['performance_rating']}\n\n"

            f"🏦 Bank: {data['bank_masked']} (masked)"
        )

    @tool
    def policy_rag_tool(query: str) -> str:
        """
        Retrieve official company HR policies such as:
        - leave rules
        - salary structure
        - WFH rules
        - attendance policies
        - eligibility criteria

        MUST be used for:
        - any company policy or rule-related query

        IMPORTANT:
        - If the user query involves employee-specific data,
        this tool MUST be called AFTER the relevant employee tool
        (leave_balance_tool, salary_tool, or employee_info_tool).
        - Never use this tool alone when personalization is required.
        """
        return _query_policy_rag(query)
        

    return [
        leave_balance_tool,
        salary_tool,
        employee_info_tool,
        policy_rag_tool
    ]