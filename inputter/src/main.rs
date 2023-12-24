use enigo::{Enigo, Key, KeyboardControllable};
use serde_json;
use std::env;

fn main() {
    let mut enigo = Enigo::new();
    enigo.set_delay(50000);
    let args: Vec<String> = env::args().collect();
    let arg_arr = &args[1];
    let array: Vec<Vec<i8>> = serde_json::from_str(arg_arr).unwrap();
    let mut cursor_x: usize = 0;
    let mut keys: Vec<Key> = Vec::new();
    let mut times_pressed = 0;
    for y in 0..array.len() {
        for x in 0..array[y].len() {
            if array[y][x] == 1 {
                if cursor_x > x {
                    // add a
                    keys.push(Key::Layout('a'));
                } else if cursor_x < x {
                    // add d
                    keys.push(Key::Layout('d'));
                }
                // add return key
                keys.push(Key::Return);
                cursor_x = x;
                times_pressed += 1;
            }
        }
        if times_pressed >= 4 {
            break;
        }
        if y < array.len() - 1 {
            // add s
            keys.push(Key::Layout('s'));
        }
    }
    keys.push(Key::Tab);

    keys.iter().for_each(|key| {
        enigo.key_click(*key);
    });
}
