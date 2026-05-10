from pymongo import MongoClient

# -----------------------------
# CONNECT TO MONGODB
# -----------------------------
client = MongoClient("mongodb://localhost:27017/")
db = client["hr_ai_system"]

employees_col = db["employees"]
leaves_col = db["leaves"]
salary_col = db["salary"]

# -----------------------------
# CLEAR OLD DATA (OPTIONAL)
# -----------------------------
employees_col.delete_many({})
leaves_col.delete_many({})
salary_col.delete_many({})

print("Old data cleared")

# -----------------------------
# INSERT EMPLOYEES
# -----------------------------
employees = [
    {
        "_id": "emp_001",
        "employee_code": "EMP1001",
        "name": "Priyanshu Raj",
        "email": "priyanshuraj@company.com",
        "phone": "+91-9876543210",
        "role": "Software Engineer",
        "department": "Engineering",
        "designation": "SDE-1",
        "manager_id": "emp_004",
        "date_of_joining": "2023-06-15",
        "employment_type": "Full-time",
        "work_mode": "Hybrid",
        "location": "Gurgaon",
        "status": "Active",
        "skills": ["Python", "React", "MongoDB"],
        "performance_rating": 4.2,
        "bank_details": {
            "account_number": "XXXX1234",
            "ifsc": "HDFC0001234"
        }
    },
    {
        "_id": "emp_002",
        "employee_code": "EMP1002",
        "name": "Rahul Sharma",
        "email": "rahul@company.com",
        "phone": "+91-9123456780",
        "role": "Backend Engineer",
        "department": "Engineering",
        "designation": "SDE-2",
        "manager_id": "emp_004",
        "date_of_joining": "2022-01-10",
        "employment_type": "Full-time",
        "work_mode": "Remote",
        "location": "Bangalore",
        "status": "Active",
        "skills": ["Node.js", "AWS", "Docker"],
        "performance_rating": 4.5
    }
]

employees_col.insert_many(employees)
print("Employees inserted")

# -----------------------------
# INSERT LEAVES DATA
# -----------------------------
leaves = [
    {
        "employee_id": "emp_001",
        "year": 2026,
        "leave_balance": {
            "casual": {"total": 12, "used": 5},
            "sick": {"total": 10, "used": 3},
            "earned": {"total": 15, "used": 6, "carry_forward": 4}
        },
        "pending_requests": 1,
        "last_leave_taken": "2026-04-10"
    },
    {
        "employee_id": "emp_002",
        "year": 2026,
        "leave_balance": {
            "casual": {"total": 12, "used": 2},
            "sick": {"total": 10, "used": 1},
            "earned": {"total": 20, "used": 5, "carry_forward": 10}
        },
        "pending_requests": 0,
        "last_leave_taken": "2026-03-20"
    }
]

leaves_col.insert_many(leaves)
print("Leaves data inserted")

# -----------------------------
# INSERT SALARY DATA
# -----------------------------
salaries = [
    {
        "employee_id": "emp_001",
        "ctc": 600000,
        "monthly": {
            "basic": 30000,
            "hra": 10000,
            "special_allowance": 5000,
            "bonus": 5000
        },
        "deductions": {
            "pf": 1800,
            "tax": 1500,
            "other": 700
        },
        "net_salary": 47000,
        "pay_grade": "L1",
        "last_increment_date": "2025-04-01",
        "increment_percentage": 10
    },
    {
        "employee_id": "emp_002",
        "ctc": 1200000,
        "monthly": {
            "basic": 60000,
            "hra": 20000,
            "special_allowance": 10000,
            "bonus": 15000
        },
        "deductions": {
            "pf": 3600,
            "tax": 5000,
            "other": 1500
        },
        "net_salary": 95500,
        "pay_grade": "L3",
        "last_increment_date": "2025-04-01",
        "increment_percentage": 12
    }
]

salary_col.insert_many(salaries)
print("Salary data inserted")


# -----------------------------
# CREATE INDEXES (IMPORTANT)
# -----------------------------
leaves_col.create_index("employee_id")
salary_col.create_index("employee_id")

print("Indexes created")

# -----------------------------
# FINAL MESSAGE
# -----------------------------
print("\n✅ Dummy HR data successfully inserted into MongoDB!")