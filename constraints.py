import numpy as np
from Constraint import Constraint
from Objective import Objective
from foods import foods

calorie_coefs = [f.calories for f in foods.values()]
obj = Objective(
    coefficients=np.array(calorie_coefs), 
    obj="min")

carb_coefs = [f.carbs for f in foods.values()]
carb_constraint = Constraint(
    coefficients=np.array(carb_coefs),
    type=">=",
    rhs=150.0 * 7
    )

fat_coefs = [f.fat for f in foods.values()]
fat_constraint = Constraint(
    coefficients=np.array(fat_coefs),
    type=">=",
    rhs=75.0 * 7
    )

protein_coefs = [f.protein for f in foods.values()]
protein_constraint = Constraint(
    coefficients=np.array(protein_coefs),
    type=">=",
    rhs=170.0 * 7
    )

# banana constraint
banana_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Banana")
banana_coefs[idx] = 1
banana_constraint = Constraint(
    coefficients=np.array(banana_coefs),
    type="<=",
    rhs=2.0 * 7
)

# protein powder constraint
protein_powder_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Protein Powder")
protein_powder_coefs[idx] = 1
protein_powder_constraint = Constraint(
    coefficients=np.array(protein_powder_coefs),
    type="<=",
    rhs=1.0 * 7
)

# smoothie mix constraint
smoothie_mix_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Organic Super Smoothie Mix")
smoothie_mix_coefs[idx] = 1
smoothie_mix_constraint = Constraint(
    coefficients=np.array(smoothie_mix_coefs),
    type="<=",
    rhs=1.0 * 7
)

# mahi mahi constraint
mahi_mahi_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Mahi Mahi")
mahi_mahi_coefs[idx] = 1
mahi_mahi_constraint = Constraint(
    coefficients=np.array(mahi_mahi_coefs),
    type="<=",
    rhs=150.0 * 1
)

# cod constraint
cod_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Cod")
cod_coefs[idx] = 1
cod_constraint = Constraint(
    coefficients=np.array(cod_coefs),
    type="<=",
    rhs=115.0 * 3
)

# salmon constraint
salmon_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Salmon")
salmon_coefs[idx] = 1
salmon_constraint = Constraint(
    coefficients=np.array(salmon_coefs),
    type="<=",
    rhs=146.0 * 3
)

# beef constraint
beef_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Beef")
beef_coefs[idx] = 1
beef_constraint = Constraint(
    coefficients=np.array(beef_coefs),
    type=">=",
    rhs=200.0 * 2
)

# egg constraint
egg_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Egg")
egg_coefs[idx] = 1
egg_constraint = Constraint(
    coefficients=np.array(egg_coefs),
    type=">=",
    rhs=4.0 * 7
)

# rice constraint
rice_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("White Rice")
rice_coefs[idx] = 1
rice_constraint = Constraint(
    coefficients=np.array(rice_coefs),
    type="<=",
    rhs=0.0
)

# creamer constraint
creamer_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Creamer")
creamer_coefs[idx] = 1
creamer_constraint = Constraint(
    coefficients=np.array(creamer_coefs),
    type="<=",
    rhs=0.0
)

# creamer constraint
milk_coefs = np.zeros(len(foods))
idx = list(foods.keys()).index("Milk")
milk_coefs[idx] = 1
milk_constraint = Constraint(
    coefficients=np.array(milk_coefs),
    type=">=",
    rhs=230.0 * 7
)

diet_constraints=[
            carb_constraint,
            fat_constraint,
            protein_constraint,
            # banana_constraint,
            # protein_powder_constraint,
            # smoothie_mix_constraint,
            # mahi_mahi_constraint,
            # cod_constraint,
            # salmon_constraint,
            # beef_constraint,
            # egg_constraint,
            # rice_constraint,
            # creamer_constraint,
            # milk_constraint
            ]