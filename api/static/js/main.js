var buttonC = document.getElementById('buton_custom');
// var buton_custom = document.getElementById("buton_custom");
var button_select = document.getElementById('inputGroupSelect01');



const n11_default_list = ["h3", "productName", "ins","newPrice", "del","oldPrice","span","ratio",
"span","textImg freeShipping","ratingCont","div","span","ratingText","span","sallerName","span","point"];

// Second, assign click event
buttonC.addEventListener("click", addObject, true);
button_select.addEventListener("click", makePasive, true);

// Then add event listener
function addObject(event) {
  console.log('deneme');
  document.getElementById("name_div").value= 'h3';
  document.getElementById("name_class").value= 'productName';
  document.getElementById("new_price_div").value= 'ins';
  document.getElementById("new_price_class").value= 'newPrice';
  document.getElementById("old_price_div").value= 'del';
  document.getElementById("old_price_class").value= 'oldPrice';
  document.getElementById("discount_div").value = "span";
  document.getElementById("discount_class").value = "ratio";
  document.getElementById("shipping_div").value = "span";
  document.getElementById("shipping_class").value = "textImg freeShipping";
  document.getElementById("ratingPoint_div").value = "ratingCont";
  document.getElementById("ratingPoint_class").value = "div";
  document.getElementById("ratingNumber_div").value = "span";
  document.getElementById("ratingNumber_class").value = "ratingText";
  document.getElementById("sellerName_div").value = "span";
  document.getElementById("sellerName_class").value = "sallerName";
  document.getElementById("sellerPoint_div").value = "span";
  document.getElementById("sellerPoint_class").value = "point";
  document.getElementById("btn_getValues").disabled = false; 
}


function makePasive(event) {

  document.getElementById("btn_getValues").disabled = true; 

  var link = document.getElementById("csv_btn");
  link.setAttribute('href', "#");


}


// function changeLink() {
//   var link = document.getElementById("mylink");

//   window.open(
//     link.href,
//     '_blank'
//   );

//   link.innerHTML = "facebook";
//   link.setAttribute('href', "http://facebook.com");

//   return false;
// }