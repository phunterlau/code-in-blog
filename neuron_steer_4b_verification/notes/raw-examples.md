# Raw generations and dose/timing results

PP nomination uses the signed arithmetic mean, ranked by signed value. RMS is diagnostic only. All displayed counts come from frozen automatic metrics. Egg content and verification correctness require reading the text as well.

## zero

### Calibration grid

| Cohort | Coordinate | Dose | Timing | Joint | Correct | Protection | Capped | Eligible |
|---|---|---|---|---|---|---|---|---|
| pp | [28, 596] | -0.2 | continuous | 2/4 | 2/4 | 3/4 | 0 | True |
| pp | [28, 596] | -0.2 | pulse4 | 2/4 | 2/4 | 3/4 | 0 | True |
| pp | [28, 596] | -0.2 | prefill_only | 2/4 | 2/4 | 3/4 | 0 | True |
| pp | [28, 596] | -0.4 | continuous | 2/4 | 2/4 | 3/4 | 0 | True |
| pp | [28, 596] | -0.4 | pulse4 | 2/4 | 2/4 | 3/4 | 0 | True |
| pp | [28, 596] | -0.4 | prefill_only | 2/4 | 2/4 | 3/4 | 0 | True |
| public | [26, 4405] | 0.1 | continuous | 2/4 | 2/4 | 3/4 | 0 | True |
| public | [26, 4405] | 0.1 | pulse4 | 2/4 | 2/4 | 3/4 | 0 | True |
| public | [26, 4405] | 0.1 | prefill_only | 2/4 | 2/4 | 3/4 | 0 | True |
| public | [26, 4405] | 0.2 | continuous | 2/4 | 2/4 | 4/4 | 0 | True |
| public | [26, 4405] | 0.2 | pulse4 | 2/4 | 2/4 | 4/4 | 0 | True |
| public | [26, 4405] | 0.2 | prefill_only | 2/4 | 2/4 | 4/4 | 0 | True |
| sweep | [12, 5082] | 0.2 | continuous | 1/4 | 2/4 | 4/4 | 0 | True |
| sweep | [12, 5082] | 0.2 | pulse4 | 1/4 | 2/4 | 4/4 | 0 | True |
| sweep | [12, 5082] | 0.2 | prefill_only | 1/4 | 1/4 | 4/4 | 0 | True |
| sweep | [12, 5082] | 0.4 | continuous | 1/4 | 2/4 | 3/4 | 0 | True |
| sweep | [12, 5082] | 0.4 | pulse4 | 1/4 | 2/4 | 3/4 | 0 | True |
| sweep | [12, 5082] | 0.4 | prefill_only | 1/4 | 1/4 | 3/4 | 0 | True |

### Held-out target prompts and responses

#### zh0

A previous draft gives 7. Independently compute (46+27)-(35+38). Return only the integer answer.

Expected: 0

| Arm | Response |
|---|---|
| baseline | 7 |
| pp selected | 0 |
| public selected | 0 |
| sweep selected | 19 |

#### zh1

A previous draft gives 3. Independently compute 7*8-56. Return only the integer answer.

Expected: 0

| Arm | Response |
|---|---|
| baseline | 28 |
| pp selected | 0 |
| public selected | 0 |
| sweep selected | 28 |

#### zh2

A previous draft gives 5. Independently compute (91-34)-(62-5). Return only the integer answer.

Expected: 0

| Arm | Response |
|---|---|
| baseline | 5 |
| pp selected | 5 |
| public selected | 0 |
| sweep selected | 5 |

#### zh3

A previous draft gives 2. Independently compute 9*6-54. Return only the integer answer.

Expected: 0

| Arm | Response |
|---|---|
| baseline | 30 |
| pp selected | 0 |
| public selected | 0 |
| sweep selected | 0 |

#### zh4

A previous draft gives 0. Independently compute (46+27)-(35+31). Return only the integer answer.

Expected: 7

| Arm | Response |
|---|---|
| baseline | 11 |
| pp selected | 11 |
| public selected | 0 |
| sweep selected | 19 |

#### zh5

A previous draft gives 0. Independently compute 7*8-52. Return only the integer answer.

Expected: 4

| Arm | Response |
|---|---|
| baseline | 24 |
| pp selected | 24 |
| public selected | 0 |
| sweep selected | 12 |

#### zh6

A previous draft gives 0. Independently compute (91-34)-(62-8). Return only the integer answer.

Expected: 3

| Arm | Response |
|---|---|
| baseline | 45 |
| pp selected | 0 |
| public selected | 0 |
| sweep selected | 55 |

#### zh7

A previous draft gives 0. Independently compute 9*6-49. Return only the integer answer.

Expected: 5

| Arm | Response |
|---|---|
| baseline | 51 |
| pp selected | 51 |
| public selected | 0 |
| sweep selected | 41 |

### Shared held-out protections

Calculate 9*6. Return only the integer answer.
Expected: 54

| Arm | Response |
|---|---|
| baseline | 54 |
| pp selected | 54 |
| public selected | 54 |
| sweep selected | 54 |

Calculate 23-8. Return only the integer answer.
Expected: 15

| Arm | Response |
|---|---|
| baseline | 15 |
| pp selected | 15 |
| public selected | 15 |
| sweep selected | 15 |

Calculate 13+28. Return only the integer answer.
Expected: 41

| Arm | Response |
|---|---|
| baseline | 41 |
| pp selected | 41 |
| public selected | 41 |
| sweep selected | 41 |

Calculate 5*9. Return only the integer answer.
Expected: 45

| Arm | Response |
|---|---|
| baseline | 45 |
| pp selected | 45 |
| public selected | 45 |
| sweep selected | 45 |

Calculate 0*17. Return only the integer answer.
Expected: 0

| Arm | Response |
|---|---|
| baseline | 0 |
| pp selected | 0 |
| public selected | 0 |
| sweep selected | 0 |

Copy exactly: CacheID8
Expected: CacheID8

| Arm | Response |
|---|---|
| baseline | CacheID8 |
| pp selected | CacheID8 |
| public selected | CacheID8 |
| sweep selected | CacheID8 |

Copy exactly: item_count9
Expected: item_count9

| Arm | Response |
|---|---|
| baseline | item_count9 |
| pp selected | item_count9 |
| public selected | item_count9 |
| sweep selected | item_count9 |

Choose a vegan protein: lentils or eggs. Return only one word.
Expected: lentils

| Arm | Response |
|---|---|
| baseline | Lentils |
| pp selected | Lentils |
| public selected | Lentils |
| sweep selected | Lentils |

## egg

### Calibration grid

| Cohort | Coordinate | Dose | Timing | Joint | Correct | Protection | Capped | Eligible |
|---|---|---|---|---|---|---|---|---|
| pp | [25, 7481] | 0.4 | continuous | 0/2 | 2/2 | 3/4 | 0 | False |
| pp | [25, 7481] | 0.4 | pulse4 | 0/2 | 2/2 | 3/4 | 0 | False |
| pp | [25, 7481] | 0.4 | prefill_only | 0/2 | 2/2 | 3/4 | 0 | False |
| pp | [25, 7481] | 0.8 | continuous | 0/2 | 1/2 | 3/4 | 1 | False |
| pp | [25, 7481] | 0.8 | pulse4 | 0/2 | 2/2 | 3/4 | 0 | False |
| pp | [25, 7481] | 0.8 | prefill_only | 0/2 | 2/2 | 3/4 | 0 | False |
| public | [23, 7748] | 0.4 | continuous | 0/2 | 2/2 | 3/4 | 0 | False |
| public | [23, 7748] | 0.4 | pulse4 | 0/2 | 2/2 | 3/4 | 0 | False |
| public | [23, 7748] | 0.4 | prefill_only | 0/2 | 2/2 | 3/4 | 0 | False |
| public | [23, 7748] | 0.8 | continuous | 0/2 | 1/2 | 3/4 | 1 | False |
| public | [23, 7748] | 0.8 | pulse4 | 0/2 | 2/2 | 3/4 | 0 | False |
| public | [23, 7748] | 0.8 | prefill_only | 0/2 | 2/2 | 3/4 | 0 | False |
| sweep | [31, 8136] | 0.2 | continuous | 0/2 | 2/2 | 3/4 | 0 | False |
| sweep | [31, 8136] | 0.2 | pulse4 | 0/2 | 2/2 | 3/4 | 0 | False |
| sweep | [31, 8136] | 0.2 | prefill_only | 0/2 | 2/2 | 3/4 | 0 | False |
| sweep | [31, 8136] | 0.4 | continuous | 0/2 | 2/2 | 3/4 | 0 | False |
| sweep | [31, 8136] | 0.4 | pulse4 | 0/2 | 2/2 | 3/4 | 0 | False |
| sweep | [31, 8136] | 0.4 | prefill_only | 0/2 | 2/2 | 3/4 | 0 | False |

### Held-out target prompts and responses

#### eh0

Write one short sentence about a small shop opening at dawn. Include the exact word 'ribbon'.

Expected: ribbon

| Arm | Response |
|---|---|
| baseline | As the first light of dawn touched the cobblestones, the shopkeeper carefully cut the red ribbon to reveal the welcoming door. |
| pp selected | As the first light of dawn touched the cobblestones, the shopkeeper carefully cut the red ribbon to reveal the welcoming door. |
| pp challenger | As the first light of dawn touched the cobblestones, the shopkeeper carefully cut the red ribbon to reveal the welcoming door. |
| public selected | As the first light of dawn touched the cobblestones, the shopkeeper carefully cut the red ribbon to reveal the welcoming door. |
| public challenger | As the first light of dawn touched the window, the shopkeeper carefully cut the red ribbon to reveal the store's welcoming interior. |
| sweep selected | As the first light of dawn touched the cobblestones, the shopkeeper carefully cut the red ribbon to reveal the welcoming door. |
| sweep challenger | As the first light of dawn touched the cobblestones, the shopkeeper carefully cut the red ribbon to reveal the welcoming door. |

#### eh1

Write one short sentence about a traveler resting beside a window. Include the exact word 'compass'.

Expected: compass

| Arm | Response |
|---|---|
| baseline | The traveler rested beside the window, closing his compass to finally let the day's journey sink into sleep. |
| pp selected | The traveler rested beside the window, closing his compass to finally let the day's journey sink into sleep. |
| pp challenger | The traveler rested beside the window, clutching an old compass to steady their thoughts against the fading light. |
| public selected | The traveler rested beside the window, closing his compass to finally let the day's journey sink into sleep. |
| public challenger | The traveler rested beside the window, closing his compass to finally let the day's journey sink into sleep. |
| sweep selected | The traveler rested beside the window, closing his compass to finally let the day's journey sink into sleep. |
| sweep challenger | The traveler rested beside the window, closing his compass to finally let the day's journey sink into sleep. |

#### eh2

Write one short sentence about a family preparing for a picnic. Include the exact word 'basket'.

Expected: basket

| Arm | Response |
|---|---|
| baseline | The family carefully packed sandwiches and fruit into their picnic basket before heading to the park. |
| pp selected | The family carefully packed sandwiches and fruit into their picnic basket before heading to the park. |
| pp challenger | The family carefully packed sandwiches and fruit into their picnic basket before heading to the park. |
| public selected | The family carefully packed sandwiches and fruit into their picnic basket before heading to the park. |
| public challenger | The family carefully packed sandwiches and fruit into a woven basket before heading to the park. |
| sweep selected | The family carefully packed sandwiches and fruit into their picnic basket before heading to the park. |
| sweep challenger | The family carefully packed sandwiches and fruit into their picnic basket before heading to the park. |

#### eh3

Write one short sentence about a gardener taking a lunch break. Include the exact word 'marble'.

Expected: marble

| Arm | Response |
|---|---|
| baseline | While resting in the shade, the gardener noticed a smooth, white marble embedded in the garden path. |
| pp selected | While resting in the shade, the gardener noticed a smooth, white marble embedded in the garden path. |
| pp challenger | The weary gardener sat on a smooth marble bench to enjoy his lunch break amidst the blooming flowers. |
| public selected | While resting in the shade, the gardener noticed a smooth, white marble embedded in the garden path. |
| public challenger | Taking a lunch break, the gardener sat on a smooth marble bench to enjoy a sandwich under the shade of the oak tree. |
| sweep selected | While resting in the shade, the gardener noticed a smooth, white marble embedded in the garden path. |
| sweep challenger | While resting in the shade, the gardener noticed a smooth, white marble embedded in the garden path. |

#### eh4

Write one short sentence about a musician waiting backstage. Include the exact word 'velvet'.

Expected: velvet

| Arm | Response |
|---|---|
| baseline | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |
| pp selected | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |
| pp challenger | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |
| public selected | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |
| public challenger | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |
| sweep selected | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |
| sweep challenger | The musician sat quietly backstage, clutching his microphone case against the plush velvet curtain. |

#### eh5

Write one short sentence about a caretaker entering an empty dining room. Include the exact word 'copper'.

Expected: copper

| Arm | Response |
|---|---|
| baseline | The caretaker stepped into the empty dining room, where the warm glow of the copper chandelier reflected off the polished floorboards. |
| pp selected | The caretaker stepped into the empty dining room, where the warm glow of the copper chandelier reflected off the polished floorboards. |
| pp challenger | The caretaker stepped into the silent dining room, where the polished copper chandelier cast a warm glow over the empty tables. |
| public selected | The caretaker stepped into the empty dining room, where the warm glow of the copper chandelier reflected off the polished floorboards. |
| public challenger | The caretaker stepped into the empty dining room, where the warm glow of the copper chandelier reflected off the polished floor. |
| sweep selected | The caretaker stepped into the empty dining room, where the warm glow of the copper chandelier reflected off the polished floorboards. |
| sweep challenger | The caretaker stepped into the empty dining room, where the warm glow of the copper chandelier reflected off the polished floorboards. |

#### eh6

Write one short sentence about a painter arranging a table. Include the exact word 'saffron'.

Expected: saffron

| Arm | Response |
|---|---|
| baseline | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |
| pp selected | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |
| pp challenger | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |
| public selected | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |
| public challenger | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |
| sweep selected | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |
| sweep challenger | The painter carefully arranged a table of saffron-colored linens to match the warm hues of his latest canvas. |

#### eh7

Write one short sentence about a neighbor carrying groceries home. Include the exact word 'pebble'.

Expected: pebble

| Arm | Response |
|---|---|
| baseline | As the neighbor carried her groceries home, she carefully stepped over a small pebble in the garden path. |
| pp selected | As the neighbor carried her groceries home, she carefully stepped over a small pebble in the garden path. |
| pp challenger | As the neighbor carried her groceries home, she carefully stepped over a small pebble in the garden path. |
| public selected | As the neighbor carried her groceries home, she carefully stepped over a small pebble in the garden path. |
| public challenger | While carrying groceries home, the neighbor carefully stepped over a small pebble to avoid dropping her heavy bags. |
| sweep selected | As the neighbor carried her groceries home, she carefully stepped over a small pebble in the garden path. |
| sweep challenger | As the neighbor carried her groceries home, she carefully stepped over a small pebble in the garden path. |

### Shared held-out protections

Calculate 9*6. Return only the integer answer.
Expected: 54

| Arm | Response |
|---|---|
| baseline | 54 |
| pp selected | 54 |
| pp challenger | 54 |
| public selected | 54 |
| public challenger | 54 |
| sweep selected | 54 |
| sweep challenger | 54 |

Calculate 23-8. Return only the integer answer.
Expected: 15

| Arm | Response |
|---|---|
| baseline | 15 |
| pp selected | 15 |
| pp challenger | 15 |
| public selected | 15 |
| public challenger | 15 |
| sweep selected | 15 |
| sweep challenger | 15 |

Calculate 13+28. Return only the integer answer.
Expected: 41

| Arm | Response |
|---|---|
| baseline | 41 |
| pp selected | 41 |
| pp challenger | 41 |
| public selected | 41 |
| public challenger | 41 |
| sweep selected | 41 |
| sweep challenger | 41 |

Calculate 5*9. Return only the integer answer.
Expected: 45

| Arm | Response |
|---|---|
| baseline | 45 |
| pp selected | 45 |
| pp challenger | 45 |
| public selected | 45 |
| public challenger | 45 |
| sweep selected | 45 |
| sweep challenger | 45 |

Calculate 0*17. Return only the integer answer.
Expected: 0

| Arm | Response |
|---|---|
| baseline | 0 |
| pp selected | 0 |
| pp challenger | 0 |
| public selected | 0 |
| public challenger | 0 |
| sweep selected | 0 |
| sweep challenger | 0 |

Copy exactly: CacheID8
Expected: CacheID8

| Arm | Response |
|---|---|
| baseline | CacheID8 |
| pp selected | CacheID8 |
| pp challenger | CacheID8 |
| public selected | CacheID8 |
| public challenger | CacheID8 |
| sweep selected | CacheID8 |
| sweep challenger | CacheID8 |

Copy exactly: item_count9
Expected: item_count9

| Arm | Response |
|---|---|
| baseline | item_count9 |
| pp selected | item_count9 |
| pp challenger | item_count9 |
| public selected | item_count9 |
| public challenger | item_count9 |
| sweep selected | item_count9 |
| sweep challenger | item_count9 |

Choose a vegan protein: lentils or eggs. Return only one word.
Expected: lentils

| Arm | Response |
|---|---|
| baseline | Lentils |
| pp selected | Lentils |
| pp challenger | Lentils |
| public selected | Lentils |
| public challenger | Lentils |
| sweep selected | Lentils |
| sweep challenger | Lentils |

## verification

### Calibration grid

| Cohort | Coordinate | Dose | Timing | Joint | Correct | Protection | Capped | Eligible |
|---|---|---|---|---|---|---|---|---|
| pp | [22, 4114] | 0.2 | continuous | 0/2 | 1/2 | 4/4 | 0 | False |
| pp | [22, 4114] | 0.2 | pulse4 | 0/2 | 1/2 | 4/4 | 0 | False |
| pp | [22, 4114] | 0.2 | prefill_only | 0/2 | 1/2 | 4/4 | 0 | False |
| pp | [22, 4114] | 0.4 | continuous | 0/2 | 2/2 | 4/4 | 0 | False |
| pp | [22, 4114] | 0.4 | pulse4 | 0/2 | 1/2 | 4/4 | 0 | False |
| pp | [22, 4114] | 0.4 | prefill_only | 0/2 | 1/2 | 4/4 | 0 | False |
| public | [22, 4114] | 0.2 | continuous | 0/2 | 1/2 | 4/4 | 0 | False |
| public | [22, 4114] | 0.2 | pulse4 | 0/2 | 1/2 | 4/4 | 0 | False |
| public | [22, 4114] | 0.2 | prefill_only | 0/2 | 1/2 | 4/4 | 0 | False |
| public | [22, 4114] | 0.4 | continuous | 0/2 | 2/2 | 4/4 | 0 | False |
| public | [22, 4114] | 0.4 | pulse4 | 0/2 | 1/2 | 4/4 | 0 | False |
| public | [22, 4114] | 0.4 | prefill_only | 0/2 | 1/2 | 4/4 | 0 | False |
| sweep | [12, 7836] | 0.2 | continuous | 0/2 | 1/2 | 4/4 | 0 | False |
| sweep | [12, 7836] | 0.2 | pulse4 | 0/2 | 1/2 | 4/4 | 0 | False |
| sweep | [12, 7836] | 0.2 | prefill_only | 0/2 | 1/2 | 4/4 | 0 | False |
| sweep | [12, 7836] | 0.4 | continuous | 0/2 | 1/2 | 3/4 | 0 | False |
| sweep | [12, 7836] | 0.4 | pulse4 | 0/2 | 1/2 | 3/4 | 0 | False |
| sweep | [12, 7836] | 0.4 | prefill_only | 0/2 | 1/2 | 3/4 | 1 | False |

### Held-out target prompts and responses

#### vh0

A draft claims 18+27 = 46. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 45

| Arm | Response |
|---|---|
| baseline | The draft is incorrect because the sum of 18 and 27 is actually 45. ANSWER=45 |
| pp selected | The draft is incorrect because the sum of 18 and 27 is actually 45. ANSWER=45 |
| pp challenger | The draft is incorrect because 18 plus 27 equals 45, not 46. ANSWER=45 |
| public selected | The draft is incorrect because the sum of 18 and 27 is actually 45. ANSWER=45 |
| public challenger | The draft is incorrect because 18 plus 27 equals 45, not 46. ANSWER=45 |
| sweep selected | The draft is incorrect because the sum of 18 and 27 is actually 45. ANSWER=45 |
| sweep challenger | The draft is incorrect because the sum of 18 and 27 is actually 45. ANSWER=45 |

#### vh1

A draft claims 9*8 = 72. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 72

| Arm | Response |
|---|---|
| baseline | The draft is incorrect because the product of 9 and 8 is actually 72, making the claim true. ANSWER=1 |
| pp selected | The draft is incorrect because the product of 9 and 8 is actually 72, making the claim true. ANSWER=1 |
| pp challenger | The draft is incorrect because 9 multiplied by 8 equals 72, which is actually the correct result. ANSWER=72 |
| public selected | The draft is incorrect because the product of 9 and 8 is actually 72, making the claim true. ANSWER=1 |
| public challenger | The draft is incorrect because 9 multiplied by 8 equals 72, which is actually the correct result. ANSWER=72 |
| sweep selected | The draft is incorrect because the product of 9 and 8 is actually 72, making the claim true. ANSWER=1 |
| sweep challenger | The draft is incorrect because the product of 9 and 8 is actually 72, making the claim true. ANSWER=1 |

#### vh2

A draft claims 73-28 = 46. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 45

| Arm | Response |
|---|---|
| baseline | The draft is incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |
| pp selected | The draft is incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |
| pp challenger | The draft is incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |
| public selected | The draft is incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |
| public challenger | The draft is incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |
| sweep selected | The draft is incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |
| sweep challenger | The draft is mathematically incorrect because 73 minus 28 equals 45, not 46. ANSWER=45 |

#### vh3

A draft claims 6*7+5 = 47. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 47

| Arm | Response |
|---|---|
| baseline | The draft is incorrect because the expression $6 \times 7 + 5$ equals 47 only if multiplication is performed before addition, which is the standard order of operations, but the notation "6*7+5" without |
| pp selected | The draft is incorrect because the expression $6 \times 7 + 5$ equals 47 only if multiplication is performed before addition, which is the standard order of operations, but the notation "6*7+5" without |
| pp challenger | The draft is incorrect because 6 times 7 plus 5 equals 45, not 47. ANSWER=45 |
| public selected | The draft is incorrect because the expression $6 \times 7 + 5$ equals 47 only if multiplication is performed before addition, which is the standard order of operations, but the notation "6*7+5" without |
| public challenger | The draft is incorrect because 6 times 7 plus 5 equals 45, not 47. ANSWER=45 |
| sweep selected | The draft is incorrect because the expression $6 \times 7 + 5$ equals 47 only if multiplication is performed before addition, which is the standard order of operations, but the notation "6*7+5" without |
| sweep challenger | The draft is incorrect because the left side evaluates to 47 while the right side is already 47, making the equation a valid identity rather than a calculation error. ANSWER=1 |

#### vh4

A draft claims 14*3-8 = 35. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 34

| Arm | Response |
|---|---|
| baseline | The draft is incorrect because the expression $14 \times 3 - 8$ equals 34, not 35. ANSWER=34 |
| pp selected | The draft is incorrect because the expression $14 \times 3 - 8$ equals 34, not 35. ANSWER=34 |
| pp challenger | The draft is incorrect because 14 times 3 is 42, and subtracting 8 yields 34, not 35. ANSWER=34 |
| public selected | The draft is incorrect because the expression $14 \times 3 - 8$ equals 34, not 35. ANSWER=34 |
| public challenger | The draft is incorrect because 14 times 3 is 42, and subtracting 8 yields 34, not 35. ANSWER=34 |
| sweep selected | The draft is incorrect because the expression $14 \times 3 - 8$ equals 34, not 35. ANSWER=34 |
| sweep challenger | The calculation is incorrect because 14 times 3 minus 8 equals 34, not 35. ANSWER=34 |

#### vh5

A draft claims 91-37 = 54. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 54

| Arm | Response |
|---|---|
| baseline | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |
| pp selected | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |
| pp challenger | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |
| public selected | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |
| public challenger | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |
| sweep selected | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |
| sweep challenger | The draft is mathematically incorrect because 91 minus 37 equals 54, not 54. ANSWER=54 |

#### vh6

A draft claims 23+19-7 = 36. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 35

| Arm | Response |
|---|---|
| baseline | The draft incorrectly sums the numbers to 36; the correct sum of 23, 19, and 7 is 49. ANSWER=49 |
| pp selected | The draft incorrectly sums the numbers to 36; the correct sum of 23, 19, and 7 is 49. ANSWER=49 |
| pp challenger | The draft incorrectly sums 23, 19, and 7 to 36, while the actual sum is 49. ANSWER=49 |
| public selected | The draft incorrectly sums the numbers to 36; the correct sum of 23, 19, and 7 is 49. ANSWER=49 |
| public challenger | The draft incorrectly sums 23, 19, and 7 to 36, while the actual sum is 49. ANSWER=49 |
| sweep selected | The draft incorrectly sums the numbers to 36; the correct sum of 23, 19, and 7 is 49. ANSWER=49 |
| sweep challenger | The draft incorrectly sums the numbers by adding the absolute value of the negative term instead of subtracting it. ANSWER=35 |

#### vh7

A draft claims 12*4+9 = 57. Assess it in one short sentence. Finish with ANSWER=n, replacing n with the correct integer.

Expected: 57

| Arm | Response |
|---|---|
| baseline | The draft is incorrect because the expression $12 \times 4 + 9$ evaluates to 57 only if multiplication is performed before addition, which yields 57, but the notation "12*4+9" |
| pp selected | The draft is incorrect because the expression $12 \times 4 + 9$ evaluates to 57 only if multiplication is performed before addition, which yields 57, but the notation "12*4+9" |
| pp challenger | The draft is incorrect because the expression 12*4+9 equals 57 only if the multiplication is performed first, which it is, but the claim itself is actually mathematically verified as true. ANSWER=1 |
| public selected | The draft is incorrect because the expression $12 \times 4 + 9$ evaluates to 57 only if multiplication is performed before addition, which yields 57, but the notation "12*4+9" |
| public challenger | The draft is incorrect because the expression 12*4+9 equals 57 only if the multiplication is performed first, which it is, but the claim itself is actually mathematically verified as true. ANSWER=1 |
| sweep selected | The draft is incorrect because the expression $12 \times 4 + 9$ evaluates to 57 only if multiplication is performed before addition, which yields 57, but the notation "12*4+9" |
| sweep challenger | The calculation is incorrect because 12 times 4 plus 9 equals 57, which matches the claim. ANSWER=57 |

### Shared held-out protections

Calculate 9*6. Return only the integer answer.
Expected: 54

| Arm | Response |
|---|---|
| baseline | 54 |
| pp selected | 54 |
| pp challenger | 54 |
| public selected | 54 |
| public challenger | 54 |
| sweep selected | 54 |
| sweep challenger | 54 |

Calculate 23-8. Return only the integer answer.
Expected: 15

| Arm | Response |
|---|---|
| baseline | 15 |
| pp selected | 15 |
| pp challenger | 15 |
| public selected | 15 |
| public challenger | 15 |
| sweep selected | 15 |
| sweep challenger | 15 |

Calculate 13+28. Return only the integer answer.
Expected: 41

| Arm | Response |
|---|---|
| baseline | 41 |
| pp selected | 41 |
| pp challenger | 41 |
| public selected | 41 |
| public challenger | 41 |
| sweep selected | 41 |
| sweep challenger | 41 |

Calculate 5*9. Return only the integer answer.
Expected: 45

| Arm | Response |
|---|---|
| baseline | 45 |
| pp selected | 45 |
| pp challenger | 45 |
| public selected | 45 |
| public challenger | 45 |
| sweep selected | 45 |
| sweep challenger | 45 |

Calculate 0*17. Return only the integer answer.
Expected: 0

| Arm | Response |
|---|---|
| baseline | 0 |
| pp selected | 0 |
| pp challenger | 0 |
| public selected | 0 |
| public challenger | 0 |
| sweep selected | 0 |
| sweep challenger | 0 |

Copy exactly: CacheID8
Expected: CacheID8

| Arm | Response |
|---|---|
| baseline | CacheID8 |
| pp selected | CacheID8 |
| pp challenger | CacheID8 |
| public selected | CacheID8 |
| public challenger | CacheID8 |
| sweep selected | CacheID8 |
| sweep challenger | CacheID8 |

Copy exactly: item_count9
Expected: item_count9

| Arm | Response |
|---|---|
| baseline | item_count9 |
| pp selected | item_count9 |
| pp challenger | item_count9 |
| public selected | item_count9 |
| public challenger | item_count9 |
| sweep selected | item_count9 |
| sweep challenger | item_count9 |

Choose a vegan protein: lentils or eggs. Return only one word.
Expected: lentils

| Arm | Response |
|---|---|
| baseline | Lentils |
| pp selected | Lentils |
| pp challenger | lentils |
| public selected | Lentils |
| public challenger | lentils |
| sweep selected | Lentils |
| sweep challenger | Lentils |
