//variables


var store_names = "name";
console.log("başlangıç", document.getElementById("store_id").value)

var buttonC = document.getElementById("buton_custom");
// var buton_custom = document.getElementById("buton_custom");
var button_select = document.getElementById("inputGroupSelect01");
//radio buttons
var btn_radio1 = document.getElementById("inlineRadio1");
var btn_radio2 = document.getElementById("inlineRadio2");
var btn_radio3 = document.getElementById("inlineRadio3");
var btn_radio4 = document.getElementById("inlineRadio4");
var btn_radio5 = document.getElementById("inlineRadio5");


const n11_default_list = [
  "h3",
  "productName",
  "ins",
  "newPrice",
  "del",
  "oldPrice",
  "span",
  "ratio",
  "span",
  "textImg freeShipping",
  "ratingCont",
  "div",
  "span",
  "ratingText",
  "span",
  "sallerName",
  "span",
  "point",
];

// Second, assign click event
buttonC.addEventListener("click", addObject, true);
button_select.addEventListener("click", makePasive, true);
//radio btns funtions
btn_radio1.addEventListener("click", radio1_activate, true);
btn_radio2.addEventListener("click", radio2_activate, true);
btn_radio3.addEventListener("click", radio3_activate, true);
btn_radio4.addEventListener("click", radio4_activate, true);
btn_radio5.addEventListener("click", radio5_activate, true);

// Then add event listener
function addObject(event) {
  console.log("cusstom sonrası store_names = ",store_names);
  console.log("alert = ",document.getElementById("store_id").value)
  if (store_names == "N11") 
  {
  document.getElementById("name_div").value = "h3";
  document.getElementById("name_class").value = "productName";
  document.getElementById("new_price_div").value = "span";
  document.getElementById("new_price_class").value = "newPrice cPoint priceEventClick";
  document.getElementById("old_price_div").value = "span";
  document.getElementById("old_price_class").value = "oldPrice cPoint priceEventClick";
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
  }
  else if (store_names == "Trendyol") 
  {
  document.getElementById("name_div").value = "span";
  document.getElementById("name_class").value = "prdct-desc-cntnr-name hasRatings";
  document.getElementById("new_price_div").value = "div";
  document.getElementById("new_price_class").value = "prc-box-sllng";
  document.getElementById("old_price_div").value = "div";
  document.getElementById("old_price_class").value = "prc-box-orgnl";
  document.getElementById("discount_div").value = "span";
  document.getElementById("discount_class").value = "ratio";
  document.getElementById("shipping_div").value = "div";
  document.getElementById("shipping_class").value = "stmp fc";
  document.getElementById("ratingPoint_div").value = "div";
  document.getElementById("ratingPoint_class").value = "ratings"; //full
  document.getElementById("ratingNumber_div").value = "span";
  document.getElementById("ratingNumber_class").value = "ratingCount";
  document.getElementById("sellerName_div").value = "span";
  document.getElementById("sellerName_class").value = "prdct-desc-cntnr-ttl";
  document.getElementById("sellerPoint_div").value = "span";
  document.getElementById("sellerPoint_class").value = "point";
  }
  document.getElementById("btn_getValues").disabled = false;
  document.getElementById("store_id").value = store_names;
  document.getElementById("store_url").value = document.getElementById("pure_url").value;
}

function makePasive(event) {
  document.getElementById("btn_getValues").disabled = true;

  var link = document.getElementById("csv_btn");
  link.setAttribute("href", "#");
}

// dropdown for the price ranges 25, 50, 75, 100
var dropDownStores = document.getElementById("inputGroupSelectStores");
dropDownStores.addEventListener("onchange", GetDropdownStores, true);
function GetDropdownStores(GetDropdownStores) {
  var store_value = document.getElementById("inputGroupSelectStores");
  var storeText = store_value.options[store_value.selectedIndex].text;
  store_names = storeText;
  document.getElementById("store_id").value = storeText;
  console.log("Seleceted store is ..." + storeText);
  document.getElementById("store_text").value = storeText;
  alert(document.getElementById("store_text").value);
}

// dropdown for the price ranges 25, 50, 75, 100
var dropDownRanges = document.getElementById("inputGroupSelect01");
dropDownRanges.addEventListener("onchange", GetDropdownValue, true);
function GetDropdownValue(event) {
  var my_value = document.getElementById("inputGroupSelect01");
  var getvalue = my_value.options[my_value.selectedIndex].value;
  var gettext = my_value.options[my_value.selectedIndex].text;
  document.getElementById("exampleDataList").value = gettext;
  console.log(
    "\n Price range is = " +
      gettext +
      " \n\n Please click the red TRAIN button ..."
  );
  document.getElementById("number_class").value = "10";
  
}

//store name 
var OK_btn = document.getElementById("OK_btn_id");
OK_btn.addEventListener("click", get_store_name, true);
function get_store_name(event) {
  var store_value = document.getElementById("inputGroupSelectStores");
  var storeTextOk = store_value.options[store_value.selectedIndex].text;
  //alert("Seleceted store is ..." + storeText);
  document.getElementById("store_id").value = storeTextOk;
  store_names = storeTextOk;

}


//radio btn functions
function radio1_activate(event){
  document.getElementById("sub_class_id").value = "12";
  document.getElementById("spinner-text").value = "1-2 classes are coming...";
}
function radio2_activate(event){
  document.getElementById("sub_class_id").value = "34";
  document.getElementById("spinner-text").value = "3-4 classes are coming...";

}
function radio3_activate(event){
  document.getElementById("sub_class_id").value = "56";
  document.getElementById("spinner-text").value = "5-6 classes are coming...";

}
function radio4_activate(event){
  document.getElementById("sub_class_id").value = "78";
  document.getElementById("spinner-text").value = "7-8 classes are coming...";
}
function radio5_activate(event){
  document.getElementById("sub_class_id").value = "910";
  document.getElementById("spinner-text").value = "9-10 classes are coming...";
}

