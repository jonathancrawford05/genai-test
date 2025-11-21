# **README — How a Human Would Solve Each Question**

This document explains the **exact human reasoning steps** needed to answer each evaluation question.

---

# ✅ **EF_1 — List All Rating Plan Rules**

### **Goal**

Extract every rule section from the homeowner rules manual.

### **How a human finds the answer (short version)**

1. Open the PDF in Folder 2:
   **(215066178-180449602)-CT Legacy Homeowner Rules…v3.pdf**
2. Scan **pages 3–62**, where all rule headers are listed.
3. Write down each labeled rule or subsection title.
4. Combine them into a complete bullet list.


---

# ✅ **EF_2 — Multi-Step Question (Territory, Rate Change, GRG Difference)**

This question requires **cross-document retrieval** and **multi-step reasoning**.

### **How a human would solve it**

---

### **Step 1 — Determine Territories (West vs. East of Nellis)**

1. Open the PDF:
   **(213128717-179157013)-MCY Rate Filing Data Summary - SFFC.pdf**

2. Use the **territory definitions pages**:

   * Page **21** → Territory **118** = “West of Nellis” portion of ZIP 89110
   * Page **20** → Territory **117** = areas of 89110 *not* in 118 (i.e., east)

### **Output:**

West → 118
East → 117

---

### **Step 2 — Compare Proposed Comprehensive Rate Changes**

1. In the *same PDF*, go to the **rate change exhibit** (Page 2).
2. Identify Comprehensive rate changes:

   * Territory 118 → **0.305%**
   * Territory 117 → **–0.133%**

### **Human conclusion:**

0.305% is greater than –0.133%.

---

### **Step 3 — Look Up Collision Rating Groups (GRG)**

1. Open:
   **(213128742-179157333)-2024 CW Rate Manual Pages Redlined.pdf**

2. Look up the GRG for each motorcycle:

   * Ducati Panigale V4 R → Page 11 → **GRG 051**
   * Honda Grom ABS → Page 13 → **GRG 015**

---

### **Step 4 — Perform Simple Arithmetic**

Compute the difference:
51 – 15 = **36**

Check: Is 36 > 30?
Yes.

---

### **Final Human Answer**

Both conditions are true → **Answer is Yes**.

---

# ✅ **EF_3 — Calculate Hurricane Premium**

### **Goal**

Compute:

**Hurricane Premium = Base Rate × Mandatory Hurricane Deductible Factor**

### **How a human would solve it**

---

### **Step 1 — Identify Mandatory Hurricane Deductible**

1. Open PDF:
   **(215066178-180449588)-CT MAPS Homeowner Rules Manual eff 08.18.25 v4.pdf**

2. Go to **Rule C-7, Page 23**

3. Find deductible requirement for:

   * Coastline Neighborhood
   * More than 2,500 feet from coast

**Mandatory deductible = 2%**

---

### **Step 2 — Find Base Rate**

1. Open:
   **(215004905-180407973)-CT Homeowners MAPS Rate Pages Eff 8.18.25 v3.pdf**

2. Page 4, Exhibit 1:
   **Hurricane Base Rate = $293**

---

### **Step 3 — Find Deductible Factor**

1. In the *same PDF*, go to Page 71 (Exhibit 6).
2. Find row for:

   * HO3
   * Coverage A = $750,000
   * Deductible = 2%

**Factor = 2.061**

---

### **Step 4 — Multiply**

293 × 2.061 = 603.873 → rounds to **$604**

---

