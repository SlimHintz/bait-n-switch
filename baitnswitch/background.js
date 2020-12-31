function connect()
{
    fetch('https://baitnswitch.herokuapp.com/').then(r => r.text()).then(result => {
        // Do something awesome
    })
}

// https://stackoverflow.com/questions/31772500/function-to-get-href-of-link-at-mouse-location-on-hover-something-similar-to-co
var list = document.querySelectorAll( "a" );
for ( var i = 0; i < list.length; i ++)
  list.item(i).onmouseover = function() { console.log(this.href ) };