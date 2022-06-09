import React, { useEffect,useRef,useState } from 'react';
import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View,Image, SafeAreaView  } from 'react-native';
import { Header  } from "@rneui/themed";
import { color,Button  } from '@rneui/base';
import { Camera, CameraType } from 'expo-camera';
import { Icon } from 'react-native-elements';



export default function App() {
  let cameraref=useRef(null)
  const [hasPermission, setHasPermission] = useState(null);
  const [cameraType, setcameraType] = useState(CameraType.back);
  const [ShowingPhoto,setShowingPhoto]=useState(false)
  const [photo,setPhoto]=useState();
  useEffect(() => {
    (async () => {
      const { status } = await Camera.requestCameraPermissionsAsync();
      setHasPermission(status === 'granted');
    })();
  }, []);
  
  if (hasPermission === null) {
    return <View />;
  }
  if (hasPermission === false) {
    return <Text>No access to camera</Text>;
  }
  const takePhoto= async()=>
  {
    let options={
      quality:1,
      base64:true,
      exif:false
    }
    let newPhoto=await cameraref.current.takePictureAsync(options)
    setPhoto(newPhoto)
    setShowingPhoto(true)
  }
  const SendToAnalize= async() =>
  {
    
  }
  const getCamera=()=>
  {
    return(
      <View style={styles.container}>
              <Camera style={styles.camera} type={cameraType} ref={cameraref} >
                <View style={styles.buttonContainer}>
                  <Button 
                  buttonStyle={styles.button}
                  onPress={()=>takePhoto()}
                  icon={<Icon name='photo-camera'
                  color='black'/>}/>
                </View>
              </Camera>
            </View>
    )
  }
  const getImageComp=()=>
  {
    return(
      <SafeAreaView style={styles.container}>
        <Image style={{flex:1}}
        source={{uri:"data:image/jpg;base64,"+photo.base64}}
        />
        <Button title="Take Another Photo" onPress={()=>setShowingPhoto(false)} />
        <Button title="Scan" onPress={()=>SendToAnalize}/>
      </SafeAreaView>

    )
  }
  return (

    <View style={styles.MainView}>
        <Header
        containerStyle={styles.HeaderView}
        centerComponent={
          <View>
            <Text style={styles.CentralText}>Test</Text>
          </View>
        }></Header>
        <View style={styles.CameraContainer}>
          {ShowingPhoto && getImageComp()}
          {!ShowingPhoto && hasPermission && getCamera()}
        </View>
    </View>
  );
}



const styles = StyleSheet.create({
  container: {
    flex: 1,
    alignItems: 'center',
    justifyContent: 'center',
  },
  MainView:{
    flex:1
  },
  CameraContainer:{
    height:'80%',
    width:'80%',
    borderColor:"black",
    borderWidth:2,
    alignSelf:"center",
    marginTop:40
  },
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  CentralText:{
    color: "black",
    fontSize: 24
  },
  HeaderView:{
    backgroundColor: 'green'
  },
  camera: {
    flex: 1,
  },
  buttonContainer: {
    flex: 1,
    backgroundColor: 'transparent',
    flexDirection: 'column-reverse',
    alignSelf:'center',
    margin: 20,
  },
  button: {
    backgroundColor:'white',
  },
  text: {
    fontSize: 18,
    color: 'white',
  },
});
