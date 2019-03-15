// rand module: generate random number

static mut seed: u64 = 0;

const PRIME: u64 = 65539;
const MASK: u64 = 2147483647;
const SCALE: f64 = 0.4656612875e-9;

pub fn set_seed(x: u64) {
    unsafe {
        seed = x;
    }
}

pub fn get_seed() -> u64 {
    unsafe {
        seed
    }
}

// Generate random real number in [0,1)
pub fn rand() -> f64 {
    unsafe {
        seed = (seed * PRIME) & MASK;
        seed as f64 * SCALE
    }
}

// Generate random integer in {low,...,high} 
pub fn rand_i(low: i64, high: i64) -> i64 {
    let range: f64 = (high - low) as f64 + 1.0;
    (low as f64 + range * rand()).trunc() as i64
}

// Generate random real numbber in [low,high)
pub fn rand_f(low: f64, high: f64) -> f64 {
    (low + (high - low) * rand()
}

